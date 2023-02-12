#Import Lib
import os
import cv2
import numpy as np

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
def openVideo(path : str, scTime : int):
    #open webcam and show in folder
    vid = cv2.VideoCapture(0)
    imgCnt = 0
    timmer = 0
    while(True):
        timmer += 1
        # Capture the video frame
        ret, frame = vid.read()
  
        # Display the resulting frame
        cv2.imshow('frame', frame)
      
        # hotkey assignment
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif timmer%scTime == 0:
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
