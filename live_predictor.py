import numpy as np
import time

class rolling_sum():
    def __init__(self, buffer_size=15):
        self.queue = []
        self.sum = None
        self.buffer_size = buffer_size
    
    # vector should be a numpy array
    def add_vector(self, vector):
        if self.sum == None:
            self.sum = vector
        while len(self.queue) >= self.buffer_size:
            self.sum -= self.queue.pop(0)
        self.sum += vector
        self.queue.append(vector)
        
    # not finished, should return a dictionary of (index, percentage)
    def get_confidences(self, top_n=3):
        return self.sum[0:top_n] / np.sum(self.sum)