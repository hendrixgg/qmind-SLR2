#Import Lib
import os
import cv2
import numpy as np
import test_model

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
    #open webcam and show in folder
    vid = cv2.VideoCapture(0)
    imgCnt = 0
    timmer = 0
    while(True):
        timmer += 1
        # Capture the video frame
        ret, frame = vid.read()

        # predict the current letter
        prediction = f"predicted letter: {test_model.predict_unformatted(frame)}" if make_predictions else "?"

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.displayOverlay('frame', f"predicted letter: {prediction}")

        # hotkey assignment
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif capture_mode and timmer % scTime == 0:
            imgName = "{}.png".format(imgCnt)
            cv2.imwrite(os.path.join(path, imgName),frame)
            imgCnt += 1
  
    # After the loop release the cap object
    vid.release()

    #Destroy all the windows
    cv2.destroyAllWindows()

#--------------------(Main)-------------------------#
def main():    
    dir = createDir('c')
    openVideo(dir, 150)

if __name__ == "__main__":
    main()
