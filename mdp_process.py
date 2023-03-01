#Import Lib
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
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
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            np.append(ldm, hand_landmarks)
    else:
        return None
    return ldm

def rawToLdm(raw_img):
    '''mnist dataset is csv
    read csv to image
    then load image in mediapipe
    then convert to landmark array / csv data'''
    image = cv2.imdecode(raw_img)
    with mp_hands.Hands(
        static_image_mode = True,
        max_num_hands = 1,
        min_detection_confidence = 0.5) as hands:
        ldm = imgToLdm(hands, image)
        return ldm


#--------------------(Main)-------------------------#
def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    path = createDir('mdp_mnist')
    # get image data from files #
    train_raw, test_raw = pd.read_csv('mnist_handsigns/sign_mnist_train.csv'), pd.read_csv('mnist_handsigns/sign_mnist_test.csv')

    train_labels, train_images = train_raw['label'].values, np.reshape(train_raw.iloc[:, 1:].values, (27455, 28, 28)) / 255.0
    test_labels, test_images = test_raw['label'].values, np.reshape(test_raw.iloc[:, 1:].values, (7172, 28, 28)) / 255.0

    train_file = open(path + "train.csv", 'w')
    writer1 = csv.writer(train_file)
    for lab_train, img_train in train_labels, train_images:
        ldm_train = rawToLdm(img_train)
        row = writer1.writerow(ldm_train)
        row.append(lab_train)
    train_file.close()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()