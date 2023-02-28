import tensorflow as tf
import numpy as np
from process_image import rescale_image, rescale_image_from_file

model = tf.keras.models.load_model('models/asl_model3')
input_shape = model.layers[0].input_shape[1:-1]

# prints the model architecture summary
def model_summary():
    model.summary()

# returns the letter label for a model output value
def get_label(integer_value):
    label = ord('A') + integer_value
    # return chr(label) if label < ord('J') else chr(label + 1)
    return chr(label)

# Takes a (28, 28) numpy array with values 0-1 and returns the model output vector
# output has shape (, 24), one entry for the confidence of each letter's prediction
def predict(img):
    # there may be a better way to do this check. could take a look at the model.predict documentation.
    if not isinstance(img, np.ndarray) or img.shape != input_shape:
        print("error in image prediction: invalid image shape")
        return "ERROR"
    return model.predict(np.asarray([img]), verbose=0)[0]

# takes an unformatted cv2 image and predicts the ASL handsign in it
# ASSUMES THE IMAGE HAS PIXEL VALUES FROM 0-255
def predict_unformatted(img):
    return predict(rescale_image(img, shape=input_shape))

# takes a path to an image file (png or file readable by cv2)
def predict_file(img_path: str):
    return predict(rescale_image_from_file(img_path, shape=input_shape))