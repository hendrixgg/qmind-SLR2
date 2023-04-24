# IMPORT PYODIDE PACKAGES 
from itertools import chain
import numpy as np

# input should be the hand landmarks for one hand as outputted from mediapipe.solutions.hands.process(frame), where frame is a 3 channel image in the RGB format
# for now the input will be just a list of the hand datapoints for each frame along with a timestamp
def preprocess(bytes, inputNames):

    # DECLARE FEEDS DICTIONARY
    feeds = dict()
    inputName = str(inputNames[0])

    # Transform the input as necessary
    # in this case that means converting the hand landmarks into a numpy array of shape (, 60)
    bytesVal = np.frombuffer(bytes)
    feedVal = np.bytesVal.astype(np.float32)

    feeds[inputName] = feedVal

    return feeds
