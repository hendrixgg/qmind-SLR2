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

# def main():
#     sliding_window = rolling_sum()

#     sliding_window.add_vector(np.asarray([1.5, 2.1, 5, 3, 2]))
#     print(sliding_window.get_confidences(3))

#     sliding_window.add_vector(np.asarray([6.5, 1.1, 0.3, -1.2, 2.6]))
#     print(sliding_window.get_confidences(3))

#     print(sliding_window.sum)

# if __name__ == "__main__":
#     main()