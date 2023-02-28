#Import Lib
import os
import cv2
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def readLandmarks (img_files):
    with mp_hands.Hands(
        static_image_mode = True,
        max_num_hand = 1,
        min_detection_confidence = 0.5) as hands:
        for idx, file in enumerate(img_files):
            #Read an image

def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    # get image data from files #
    train_set, test_set = pd.read_csv('mnist_handsigns/sign_mnist_train.csv'), pd.read_csv('mnist_handsigns/sign_mnist_test.csv')

    train_labels, train_images = train_set['label'].values, np.reshape(train_set.iloc[:, 1:].values, (27455, 28, 28)) / 255.0
    test_labels, test_images = test_set['label'].values, np.reshape(test_set.iloc[:, 1:].values, (7172, 28, 28)) / 255.0

    img = train_images[0]
    raw_img = cv2.imread("LetterData/c/c0.png")
    print(raw_img)
    print(raw_img.shape)