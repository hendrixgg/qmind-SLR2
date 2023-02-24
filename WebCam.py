#Import Lib
import os
import cv2
import asl_model
import live_predictor
import mediapipe as mp
#------------------(Functions)----------------------#

#greyscale images func Input(img.jpg) --> output(img.jpg)
def greyScale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#ScreenGrab an image from video

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
    rolling_prediction = live_predictor.rolling_sum(buffer_size=15)

    mp_drawing = mp.solutions.drawing_utils
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    
    #open webcam and show in folder
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w, _ = frame.shape

    imgCnt = 0
    timmer = 0

    while(True):
        timmer += 1
        # Capture the video frame
        ret, frame = cap.read()
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get the hand landmarks
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks

        x_min, y_min, x_max, y_max = w,h,0,0
        # Iterate through the hand landmarks to get the boundaries for bounding box
        if hand_landmarks:
            for handLMs in hand_landmarks:
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv2.rectangle(frame, (x_min-50, y_min-50), (x_max+30, y_max+30), (0, 255, 0), 2)

                #turn this statement on to show the display of landmarks
                #mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)

        # cropped the video frame according to the bounding box
        if y_min <= y_max and x_min <= x_max:
            cropped = frame[y_min-50:y_max+30, x_min-50:x_max+30]
        else:
            cropped = frame

        # predict the current letter
        if make_predictions:
            try:
                rolling_prediction.add_vector(asl_model.predict_unformatted(cropped))
            except:
                print(f"{cropped}, {cropped.shape=}")
                exit(0)
        # get the top 3 predictions
            predictions = [(asl_model.get_label(i), c) for (i, c) in rolling_prediction.get_confidences(3)]
        else:
            predictions = "?"
        # if there has been low confidence in gestures, treat the situation as a "transition between gestures" and perhaps add the previouly predicted letter to a string to record the interpretations
        # if a "transition" was the previous "letter" do not add anything to the string recording the interpretation
        # to make the framerate not be so slow, could find a way to make this asyncronus


        # Display the resulting frame
        cv2.putText(frame, f"predicted letter: {predictions}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(209,80,0,255),3)
        cv2.imshow('frame', frame)
        #cv2.displayOverlay('frame', f"predicted letter: {predictions}")
        # the overlay could include more information
        # the openVideo function could take in some more parameters, giving the option to show more data on overlay

        # hotkey assignment
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif capture_mode and timmer % scTime == 0:
            imgName = f"{imgCnt}.png"
            cv2.imwrite(os.path.join(path, imgName),cropped)
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