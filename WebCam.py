#Import Lib
import os
import cv2
import numpy as np

#------------------(Functions)----------------------#

#greyscale images func Input(img.jpg) --> output(img.jpg)
def greyScale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#ScreenGrab an image from video


#Create folder to save images to
def saveImg(letter: chr):
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


    


    

#video Feed
def openVideo():
    #open webcam and show in folder
    vid = cv2.VideoCapture(0)
  
    while(True):
      
        # Capture the video frame
        ret, frame = vid.read()
  
        # Display the resulting frame
        cv2.imshow('frame', frame)
      
        # to exit program 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # After the loop release the cap object
    vid.release()

    #Destroy all the windows
    cv2.destroyAllWindows()

#--------------------(Main)-------------------------#
saveImg('a')
openVideo()
