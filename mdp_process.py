#Import Lib
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pipeline
from PIL import Image

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def createDir(letter: chr):
    #get directory of python file and add directory of the letter folder
    dir = os.getcwd()
    newDir = "LetterData\\" + letter
    dir = os.path.join(dir, newDir)

    #try creating the folder if it doent already exist
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error creating directory" + dir)
    
    return dir

def imgToLdm(hands, image):
    '''input: hands = define hand utility; image of one single hand (jpeg?)
    output: an (21,3) array of landmark data
    one landmark = [x, y, z]'''
    ldm = np.empty((21,3))
    results = hands.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            np.append(ldm, hand_landmarks)
    return ldm


def rawToLdm(image):
    #load the image in cv2
    with mp_hands.Hands(
        static_image_mode = True,
        max_num_hands = 1,
        min_detection_confidence = 0.5) as hands:
        #detect hand landmarks using mediapipe
        ldm = imgToLdm(hands, image.astype(np.uint8))
        return ldm


#--------------------(Main)-------------------------#
def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    
    train_set, test_set = pd.read_csv('mnist_handsigns/sign_mnist_train.csv'), pd.read_csv('mnist_handsigns/sign_mnist_test.csv')

    train_labels, train_images = train_set['label'].values, np.reshape(train_set.iloc[:, 1:].values, (27455, 28, 28)) / 255.0
    test_labels, test_images = test_set['label'].values, np.reshape(test_set.iloc[:, 1:].values, (7172, 28, 28)) / 255.0

    path = createDir('ldm_mnist_train')
    file = open(path + '.csv', 'w')
    writer = csv.writer(file)
    for trimg in train_images:
        ldm = rawToLdm(trimg)
        writer.writerow(ldm)
    file.close()
    
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()