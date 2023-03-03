#Import Lib
import os
import cv2
import numpy as np
import csv
import live_predictor
import mediapipe as mp
import mdp_process
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

'''Takes camera input and write hand landmarks into a csv file, 
each line represents a hand, with each list in a row represents a landmark. 
Frames without a hand would be ignored, but hand can be partially covered and landmarks can be nan.'''

def openVideo(path : str="", scTime : int=0, make_predictions: bool=True):
    capture_mode = False if path == "" or scTime == 0 else True

    file = open(path + "\mdp.csv", 'w')
    writer = csv.writer(file)

    # logic for live language recognition
    rolling_prediction = live_predictor.rolling_sum(buffer_size=15)
    text_prediction = live_predictor.text_builder(time_interval=500)

    imgCnt = 0
    timmer = 0
    
    # open webcam
    cap = cv2.VideoCapture(0)

    # hand landmarks utility
    with mp_hands.Hands(
        model_complexity = 0,
        max_num_hands = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as hands:
        while (True):
            timmer += 1
            sucess, image = cap.read()
            ldm_data = np.empty((21,3))
            
            if not sucess:
                # there is no hand in the frame
                # reset the rolling prediction
                rolling_prediction.reset()
                # treat the scenario as a space between words
                text_prediction.update(' ')
                print("empty frame")
                continue
            else: 
                ldm_data = mdp_process.imgToLdm(hands,image)

            cv2.imshow('frame', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif capture_mode and timmer % scTime == 0:
                writer.writerow(ldm_data)
                ldm_data = np.empty((21,3))
                imgCnt += 1
                continue

    cap.release()
    cv2.destroyAllWindows()
    file.close()

#--------------------(Main)-------------------------#
def main():
    dir = mdp_process.createDir('mdp')
    openVideo(dir, 150, make_predictions=False)

if __name__ == "__main__":
    main()