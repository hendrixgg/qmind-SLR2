import tensorflow as tf
import numpy as np
from enum import Enum
from process_image import rescale_image, rescale_image_from_file

class MODEL_INPUT(Enum):
    # signifies that the model takes an image as an input
    IMAGE = 1
    # signifies that the model takes in 21 3D hand landmark points from the mediapipe hands model in the form of an array of length 63
    MP_LANDMARKS = 2

model = tf.keras.models.load_model('models/asl_model2')
input_shape = model.layers[0].input_shape[1:]
model_input = MODEL_INPUT.IMAGE if len(input_shape.shape) > 1 else MODEL_INPUT.MP_LANDMARKS

# prints the model architecture summary
def model_summary():
    model.summary()

# returns the letter label for a model output value
def get_label(integer_value, include_J=False):
    label = ord('A') + integer_value
    return chr(label) if include_J or label < ord('J') else chr(label + 1)

# Takes a (28, 28) numpy array with values 0-1 and returns the model output vector
# output has shape (, 24), one entry for the confidence of each letter's prediction
def predict(input):
    # there may be a better way to do this check. could take a look at the model.predict documentation.
    if not isinstance(input, np.ndarray) or input.shape != input_shape:
        print("error in image prediction: invalid image shape")
        return "ERROR"
    return model.predict(np.asarray([input]), verbose=0)[0]

# takes an unformatted cv2 image and predicts the ASL handsign in it
# ASSUMES THE IMAGE HAS PIXEL VALUES FROM 0-255
def predict_unformatted(input):
    return predict(rescale_image(input, shape=input_shape))

# takes a path to an image file (png or file readable by cv2)
def predict_file(input_path: str):
    return predict(rescale_image_from_file(input_path, shape=input_shape))