# IMPORT PYODIDE PACKAGES 
from itertools import chain
import numpy as np

# input should be the hand landmarks for one hand as outputted from mediapipe.solutions.hands.process(frame), where frame is a 3 channel image in the RGB format
def preprocess(bytes, inputNames):

    # DECLARE FEEDS DICTIONARY
    feeds = dict()
    inputNames = str(inputNames[0])

    # Transform the input as necessary
    # in this case that means converting the hand landmarks into a numpy array of shape (, 60)
    bytesInput = np.frombuffer(bytes)
    npBytes = 
