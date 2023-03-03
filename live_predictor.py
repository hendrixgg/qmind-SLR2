import numpy as np
import time

# the sum could be implemented as a priority queue for better performance
# I am not sure if min_significance is the way to go
class rolling_sum():
    def __init__(self, buffer_size=15, min_significance=0.6):
        self.queue = []
        self.sum = None
        self.buffer_size = buffer_size
        self.min_significance = min_significance

    # resets the current sum
    def reset(self):
        if not isinstance(self.sum, np.ndarray):
            return
        self.sum.fill(0)
        self.queue.clear()
    
    # vector should be a numpy array
    # adds the input vector to self.sum and appends it to the queue so it can be kept track of up to the point where it is removed
    # removes vectors from the queue if it exceeds the buffer_size
    def add_vector(self, vector):
        assert(isinstance(vector, np.ndarray))
        if np.max(vector) < self.min_significance:
            return
        self.queue.append(vector)
        if not isinstance(self.sum, np.ndarray):
            self.sum = vector
            return
        assert(self.sum.shape == vector.shape)
        self.sum = np.add(self.sum, vector)
        while len(self.queue) > self.buffer_size:
            self.sum = np.subtract(self.sum, self.queue.pop(0))
        
    # returns a list of (index, confidence) of the higest confidence predictions
    def get_topn(self, top_n=3):
        if not isinstance(self.sum, np.ndarray):
            return [(0, 0)]
        # gets the indexes of the top n confidences in the self.sum vector
        top_indexes = np.argpartition(self.sum, -top_n)[-top_n:]
        # sorts the indexes of the confidences in order of increasing confidence value
        top_indexes = top_indexes[np.argsort(self.sum[top_indexes])]
        total = np.sum(self.sum)
        total = 1 if total == 0 else total
        return [(i, self.sum[i] / total) for i in reversed(top_indexes)]

# a class that maintains state and returns whether or not the state has persisted for the desired time_interval upon an update
class live_state():
    # init_state must be comparable, time_interval is in milliseconds
    # if time_interval is less than 0, take as infinity
    def __init__(self, init_state=None, time_interval: int=500):
        self.curr_time = time.time_ns()
        self.curr_state = init_state
        self.time_interval = time_interval * 1_000_000

    # return value: True if enough time has elapsed with the same state
    def update(self, new_state):
        if self.time_interval < 1:
            return False
        time_diff = time.time_ns() - self.curr_time
        elapsed = new_state == self.curr_state and time_diff >= self.time_interval
        if elapsed or self.curr_state != new_state:
            self.curr_state = new_state
            self.curr_time += time_diff
        return elapsed


# uses the live state class to take input
# only adds a charater to the string when the input_state changes
# will add a repeat character if the time required for repeat to be inputted has elapsed
class text_builder():
    def __init__(self, init_letter: str='', time_interval: int=500, repeat_interval: int=-1, min_confidence=0.6):
        self.input_state = live_state(init_letter, time_interval)
        self.repeat_state = live_state(init_letter, repeat_interval)
        self.string = ""
        self.prev_letter = init_letter
        self.min_confidence = min_confidence

    # updates the string if the input triggers an update
    # updates are triggered if the input changes
    # returns True if updated, otherwise False
    def update(self, letter: str, confidence: float):
        if confidence < self.min_confidence:
            return False
        should_update = self.input_state.update(letter)
        should_update = should_update if letter != self.prev_letter else False
        should_repeat = self.repeat_state.update(letter)
        if should_update or should_repeat:
            self.string += letter
            self.prev_letter = letter
            return True
        return False
    
import cv2
import asl_model

class live_asl_model():
    def __init__(self):
        self.model = asl_model.Model(includeJ=True, static_image_mode=False, saved_model_path="models/svm_landmark_model.sav", use_pickle=True)
        # logic for live language recognition
        self.rolling_prediction = rolling_sum(buffer_size=15)
        self.text_prediction = text_builder(time_interval=500)
        # state variables
        self.cropped_image = None
        self.using_images = self.model.model_input_type == asl_model.MODEL_INPUT_TYPE.IMAGE
    
    # parameters:
    # - frame: image to be processed by the model
    # - top_n: number of the best predictions to be returned
    # returns: a cropped image of the hand if there was one in the image,
    # the top_n running model predictions, current text from text_builder
    def process(self, frame, make_predictions=True, overlay_bounding_box=True, overlay_landmarks=True, top_n=3):
        if make_predictions:
            success, output = self.model.predict_unformatted(frame)
        else:
            success, output = False, "[Not Predicting]"

        if not success:
            # there is no hand in the frame
            # reset the rolling prediction
            self.rolling_prediction.reset()
            # treat the scenario as a space between words
            self.text_prediction.update(' ', 1)
            return False, None, output, self.text_prediction.string

        self.cropped_image, top, bottom, hand_landmarks = self.model.get_recent_crop_square()
        if overlay_bounding_box:
            cv2.rectangle(frame, top, bottom, (0, 255, 0), 2)
        if overlay_landmarks:
            asl_model.mp_drawing.draw_landmarks(frame, hand_landmarks, asl_model.mp_hands.HAND_CONNECTIONS)

        # get the model prediction for this frame and add it to the current rolling sum of predictions
        self.rolling_prediction.add_vector(output)
        print(output)
        # get the top 3 predictions
        predictions = [(self.model.get_label(i), c) for (i, c) in self.rolling_prediction.get_topn(top_n)]
        # add predicted letter and confidence to text input
        self.text_prediction.update(*predictions[0])
        
        return True, self.cropped_image, predictions, self.text_prediction.string
        


# def main():
#     sliding_window = rolling_sum()

#     sliding_window.add_vector(np.asarray([1.5, 2.1, 5, 3, 2]))
#     print(sliding_window.get_confidences(3))

#     sliding_window.add_vector(np.asarray([6.5, 1.1, 0.3, -1.2, 2.6]))
#     print(sliding_window.get_confidences(3))

#     print(sliding_window.sum)

# if __name__ == "__main__":
#     main()