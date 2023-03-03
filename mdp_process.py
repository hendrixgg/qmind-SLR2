#Import Lib
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pipeline
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


def rawToLdm(raw_img, path):
    #save csv data into a image
    plt.imsave(path,raw_img)
    #load the image in cv2
    image = cv2.imread(path, flags = cv2.IMREAD_COLOR)
    with mp_hands.Hands(
        static_image_mode = True,
        max_num_hands = 1,
        min_detection_confidence = 0.5) as hands:
        #detect hand landmarks using mediapipe
        ldm = imgToLdm(hands, image)
        #delete the temporary image file after use
        os.remove(path)
        return ldm


#--------------------(Main)-------------------------#
def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    # get image data from files #
    model = pipeline.cnn_tf()
    model.load('train','mnist_handsigns/sign_mnist_train.csv')
    model.load('test','mnist_handsigns/sign_mnist_test.csv')
    #use same preprocessing methods
    model.pre_process1()

    print("Now converting images to landmarks:")
    temp = "temp.png"
    train_ldm = np.empty((20,21,3))
    for trimg in model.train_set[20]:
        ldm = rawToLdm(trimg, temp)
        np.append(train_ldm,ldm)
    print(train_ldm[0])
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()