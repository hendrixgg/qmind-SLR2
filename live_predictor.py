import numpy as np
import time

class rolling_sum():
    def __init__(self, init_vector, buffer_size=15):
        self.queue = []
        self.sum = init_vector
        self.buffer_size=buffer_size
    
    # vector should be a numpy array
    def add_vector(self, vector):
        while len(self.queue) >= self.buffer_size:
            self.sum -= self.queue.pop(0)
        self.sum += vector
        self.queue.append(vector)
        
    def get_confidences(self, top_n=3):
        return self.sum[0:top_n] / np.sum(self.sum)