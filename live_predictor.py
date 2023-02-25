import numpy as np
import time

class rolling_sum():
    def __init__(self, buffer_size=15):
        self.queue = []
        self.sum = None
        self.buffer_size = buffer_size
    
    # vector should be a numpy array
    def add_vector(self, vector):
        if not isinstance(self.sum, np.ndarray):
            self.sum = vector
        else:
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

class live_state():
    # init_state must be comparable, time_interval is in milliseconds
    def __init__(self, init_state=None, time_interval: int=500):
        self.curr_time = time.time_ns()
        self.curr_state = init_state
        self.time_interval = time_interval * 1_000_000

    # return value: (True if time_interval has elapsed), prev_state
    def update(self, new_state):
        new_time = time.time_ns()
        prev_state = self.curr_state
        elapsed = (new_time - self.curr_time >= self.time_interval)
        if elapsed or self.curr_state != new_state:
            self.curr_state = new_state
            self.curr_time = new_time
        return elapsed, prev_state


# uses the live state class to take input
# only adds a charater to the string when the input_state changes
class text_builder():
    def __init__(self, init_letter: str='', time_interval: int=500):
        self.input_state = live_state(init_letter, time_interval)
        self.string = ""
        self.prev_letter = init_letter

    # updates the string if the input triggers an update
    # updates are triggered if the input changes
    # returns True if updated, otherwise False
    def update(self, letter_input):
        should_update, letter = self.input.update(letter_input)
        if should_update and self.prev_letter != letter:
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