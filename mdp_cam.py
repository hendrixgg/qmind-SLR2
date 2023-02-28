#Import Lib
import os
import cv2
import numpy as np
import csv
import live_predictor
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


def openVideo(path : str="", scTime : int=0, make_predictions: bool=True):
    capture_mode = False if path == "" or scTime == 0 else True

    file = open(path + "mdp.csv", 'w')
    writer = csv.writer(file)

    # logic for live language recognition
    rolling_prediction = live_predictor.rolling_sum(buffer_size=15)
    text_prediction = live_predictor.text_builder(time_interval=500)

    imgCnt = 0
    timmer = 0
    
    # open webcam
    cap = cv2.VideoCapture(0)

    # hand bounding box utility
    with mp_hands.Hands(
        model_complexity = 0,
        max_num_hands = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as hands:
        while (True):
            timmer += 1
            sucess, image = cap.read()
            ldm_data = []
            if not sucess:
                # there is no hand in the frame
                # reset the rolling prediction
                #rolling_prediction.reset()
                # treat the scenario as a space between words
                #text_prediction.update(' ')
                print("empty frame")
                continue
            else: 
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)
                        ldm = np.array(hand_landmarks)
                        ldm_data.append(ldm)
            '''
            image.flags.writable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            #Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            '''

            cv2.imshow('frame', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif capture_mode and timmer % scTime == 0:
                writer.writerow(ldm_data)
                ldm_data = []
                imgCnt += 1
                continue

    cap.release()
    cv2.destroyAllWindows()
    file.close()

#--------------------(Main)-------------------------#
def main():
    dir = createDir('mdp')
    openVideo(dir, 150, make_predictions=False)

if __name__ == "__main__":
    main()