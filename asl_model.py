from enum import Enum
from itertools import chain
import tensorflow as tf
import numpy as np
import cv2
import pickle
from process_image import rescale_image, crop_square
import sklearn
from sklearn.preprocessing import LabelBinarizer
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class MODEL_INPUT_TYPE(Enum):
    # signifies that the model takes an image as an input
    IMAGE = 1
    # signifies that the model takes in 3D hand landmark points from the mediapipe hands model in the form of a one-dimesional array
    MP_LANDMARKS = 2

class MODEL_TYPE(Enum):
    # Model is a tensorflow.keras.models.Sequential model
    KERAS_SEQUENTIAL = 1
    # Model is a sklearn.svm.SVC model
    # must have the property: probability=True on creation
    SCIKITLEARN_SVC = 2

class Model():
    def __init__(self, includeJ=False, static_image_mode=False, saved_model_path='models/asl_model2', use_pickle=False):
        self.model = pickle.load(open(saved_model_path, "rb")) if use_pickle else tf.keras.models.load_model(saved_model_path)
        # get the type of model and input shape
        if isinstance(self.model, tf.keras.models.Sequential):
            self.model_type = MODEL_TYPE.KERAS_SEQUENTIAL
            self.input_shape = self.model.layers[0].input_shape[1:]
        elif isinstance(self.model, sklearn.svm.SVC):
            self.model_type = MODEL_TYPE.SCIKITLEARN_SVC
            self.input_shape = self.model.shape_fit_[1:]
            
        self.includeJ = includeJ
        # determine input type
        if len(self.input_shape) > 1:
            # remove channel dimesion, assuming grayscale
            self.input_shape = self.input_shape[:-1]
            self.model_input_type = MODEL_INPUT_TYPE.IMAGE
        else:
            self.model_input_type = MODEL_INPUT_TYPE.MP_LANDMARKS

        # data from last cropped image
        self.cropped = None
        self.top_left_crop_point = None
        self.bottom_right_crop_point = None
        
        # mediapipe resources
        self.hands = mp_hands.Hands(max_num_hands=1, static_image_mode=static_image_mode)
        self.hand_landmarks = None

    # prints the model architecture summary
    def model_summary(self):
        if self.model_type == MODEL_TYPE.KERAS_SEQUENTIAL:
            self.model.summary()
        elif self.model_type == MODEL_TYPE.SCIKITLEARN_SVC:
            print(f"sklearn.svm.SVC:")
            print(f"{self.model.shape_fit_=}")
            print(f"{self.model.classes_=}")
            print(f"{self.model.n_features_in_=}")

    # returns the letter label for a model output value
    def get_label(self, integer_value):
        label = ord('A') + integer_value
        return chr(label) if self.includeJ or label < ord('J') else chr(label + 1)
    
    # returns the results from the last predict_unformatted
    def get_recent_crop_square(self):
        return self.cropped, self.top_left_crop_point, self.bottom_right_crop_point, self.hand_landmarks
    

    # Takes in formatted model input
    # output has shape (, x), one entry for the confidence of each letter's prediction
    def predict(self, model_input):
        # there may be a better way to do this check. could take a look at the model.predict documentation.
        if not isinstance(model_input, np.ndarray) or model_input.shape != self.input_shape:
            print(f"Error in model input: invalid input shape. {model_input.shape} != {self.input_shape}")
            return "ERROR"
        if self.model_type == MODEL_TYPE.KERAS_SEQUENTIAL:
            return self.model.predict(np.asarray([model_input]), verbose=0)[0]
        elif self.model_type == MODEL_TYPE.SCIKITLEARN_SVC:
            print(f"{model_input=}")
            return self.model.predict_proba([model_input])[0]

    # takes an unformatted cv2 BGR image and predicts the ASL handsign in it
    # if there is no hand found in the frame: return False, "[no hand in image]"
    # if there is a hand in the frame: return True, model output
    # ASSUMES THE IMAGE HAS PIXEL VALUES FROM 0-255
    def predict_unformatted(self, frame):
        raw_hand_landmarks = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks
        if not raw_hand_landmarks:
            return False, "[no hand in frame]"
        for h in raw_hand_landmarks:
            self.hand_landmarks = h
        self.cropped, self.top_left_crop_point, self.bottom_right_crop_point = crop_square(frame, self.hand_landmarks.landmark)
        # return the model prediction if the model was trained on images
        if self.model_input_type == MODEL_INPUT_TYPE.IMAGE:
            return True, self.predict(rescale_image(self.cropped, shape=self.input_shape))
        # wrist point
        w = self.hand_landmarks.landmark[0]
        # all of the points coordinates in a one dimesional array, with the wrist point at the origin
        scale_factor = self.cropped.shape[1] * 1. / self.cropped.shape[0]
        in_line = np.array([*chain(*chain([(p.x - w.x) * scale_factor, p.y - w.y, p.z - w.z] for p in self.hand_landmarks.landmark[1:]))])
        return True, self.predict(in_line)
    
    # the following functions typically make more sense to use when static_image_mode=True

    # takes a path to an image file (png or file readable by cv2)
    def predict_image_file(self, input_path: str):
        return self.predict_unformatted(cv2.imread(input_path))
    
    # returns the hand landmarks array generated from a file
    def produce_hand_landmarks_from_file(self, input_path):
        raw_hand_landmarks = self.hands.process(cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)).multi_hand_landmarks
        for h in raw_hand_landmarks:
            self.hand_landmarks = h
        return self.hand_landmarks.landmark

# print(np.array([*chain(*chain([a[0], a[1], a[2]] for a in [[1, 2, 3], [4, 5, 6]]))]))