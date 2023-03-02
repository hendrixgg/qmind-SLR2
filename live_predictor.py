import numpy as np
import time

# the sum could be implemented as a priority queue for better performance
class rolling_sum():
    def __init__(self, buffer_size=15):
        self.queue = []
        self.sum = None
        self.buffer_size = buffer_size

    # resets the current sum
    def reset(self):
        if not isinstance(self.sum, np.ndarray):
            return
        self.sum.fill(0)
        self.queue.clear()
    
    # vector should be a numpy array
    def add_vector(self, vector):
        assert(isinstance(vector, np.ndarray))
        if not isinstance(self.sum, np.ndarray):
            self.sum = vector
        else:
            assert(self.sum.shape == vector.shape)
            self.sum = np.add(self.sum, vector)
        while len(self.queue) >= self.buffer_size:
            self.sum = np.subtract(self.sum, self.queue.pop(0))
        self.queue.append(vector)
        
    # returns a list of (index, confidence) of the higest confidence predictions
    def get_confidences(self, top_n=3):
        # gets the indexes of the top n confidences in the self.sum vector
        top_indexes = np.argpartition(self.sum, -top_n)[-top_n:]
        # sorts the indexes of the confidences in order of increasing confidence value
        top_indexes = top_indexes[np.argsort(self.sum[top_indexes])]
        total = np.sum(self.sum)
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
    def __init__(self, init_letter: str='', time_interval: int=500, repeat_interval: int=-1, min_confidence=0.5):
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
    

# def main():
#     sliding_window = rolling_sum()

#     sliding_window.add_vector(np.asarray([1.5, 2.1, 5, 3, 2]))
#     print(sliding_window.get_confidences(3))

#     sliding_window.add_vector(np.asarray([6.5, 1.1, 0.3, -1.2, 2.6]))
#     print(sliding_window.get_confidences(3))

#     print(sliding_window.sum)

# if __name__ == "__main__":
#     main()