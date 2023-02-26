#Import Lib
import os
import cv2
import asl_model
import live_predictor
import process_image
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#------------------(Functions)----------------------#

#greyscale images func Input(img.jpg) --> output(img.jpg)
def greyScale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create folder to save images to and saves directory
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


#video Feed
# if path == "" or scTime == 0 then frames will not be saved
def openVideo(path : str="", scTime : int=0, make_predictions: bool=True):
    capture_mode = False if path == "" or scTime == 0 else True

    # logic for live language recognition
    rolling_prediction = live_predictor.rolling_sum(buffer_size=15)
    text_prediction = live_predictor.text_builder(time_interval=500, repeat_interval=5_000)


    # hand bounding box utility
    hands = mp_hands.Hands(max_num_hands=1)
    
    # open webcam
    cap = cv2.VideoCapture(0)

    imgCnt = 0
    timmer = 0

    while(True):
        timmer += 1
        # Capture the video frame
        success, frame = cap.read()
        if not success:
            print("ignoring empty camera frame.")
            continue

        # Get the hand landmarks
        hand_landmarks = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_hand_landmarks

        # Iterate through the hand landmarks to get the bounding box
        if hand_landmarks:
            # will only iterate once since only one hand is tracked at a time
            for handLMs in hand_landmarks:
                cropped, top, bottom = process_image.crop_square(frame, handLMs.landmark, 40, 40, 40, 40)
            cv2.imwrite("temp_img.png", cropped)
            cv2.rectangle(frame, top, bottom, (0, 255, 0), 2)
            # uncomment this line to display hand landmarks (joints)
            mp_drawing.draw_landmarks(frame, handLMs, mp_hands.HAND_CONNECTIONS)
        else:
            # there is no hand in the frame
            # reset the rolling prediction
            rolling_prediction.reset()
            # treat the scenario as a space between words
            text_prediction.update(' ')

        # predict the current letter
        if make_predictions and hand_landmarks:
            rolling_prediction.add_vector(asl_model.predict_unformatted(cropped))
            # get the top 3 predictions
            predictions = [(asl_model.get_label(i), c) for (i, c) in rolling_prediction.get_confidences(3)]
            # add predicted letter to text input
            text_prediction.update(predictions[0][0])
        else:
            predictions = "[not predicting]"
        # if there has been low confidence in gestures, treat the situation as a "transition between gestures" and perhaps add the previouly predicted letter to a string to record the interpretations
        # if a "transition" was the previous "letter" do not add anything to the string recording the interpretation
        # to make the framerate not be so slow, could find a way to make this asyncronus

        # Display the resulting frame
        cv2.putText(frame, f"predicted letter: {predictions}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209,80,0,255), 3)
        # if hand_landmarks:
        #     cv2.rectangle(frame, top, bottom, (0, 255, 0), 2)
        #     # uncomment this line to display hand landmarks (joints)
        #     # mp_drawing.draw_landmarks(frame, handLMs, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('frame', frame)
        # cv2.displayOverlay('frame', f"predicted letter: {predictions}")
        # the overlay could include more information
        # the openVideo function could take in some more parameters, giving the option to show more data on overlay

        # print text to console
        os.system("cls")
        print("current text:\n", text_prediction.string)

        # hotkey assignment
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif capture_mode and timmer % scTime == 0:
            imgName = f"{imgCnt}.png"
            cv2.imwrite(os.path.join(path, imgName), cropped if not any(d == 0 for d in cropped.shape) else frame)
            imgCnt += 1
  
    # After the loop release the cap object
    cap.release()

    #Destroy all the windows
    cv2.destroyAllWindows()

#--------------------(Main)-------------------------#
def main():
    dir = createDir('c')
    openVideo(dir, 150, make_predictions=False)

if __name__ == "__main__":
    main()