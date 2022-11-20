#Import Lib
import os
import cv2
import numpy as np

#------------------(Functions)----------------------#

#greyscale images func Input(img.jpg) --> output(img.jpg)
def greyScale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#ScreenGrab an image from video


#video Feed & save webcam video as images
def openVideo():
    #open webcam and show in folder
    vid = cv2.VideoCapture(0)
    #set webcam to the width and height the input camera supports
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    dir = "qmind-SLR"
    name = "asl_webcam"
    #create a folder to save the video frames
    os.mkdir(dir)
    base_path = os.path.join(dir, name)
    
    time = 0
    while vid.isOpened():

        # Capture the video frame
        ret, frame = vid.read()

        if ret == False:
            print("Can't receive frame (stream end?. Exiting ...")
            break
        
        #flip the vide frame
        frame = cv2.flip(frame, 0)

        #save the video frame as images
        if time % 50 == 0:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(time), 'jpg'), frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # to exit program 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time +=1
  
    # After the loop release the cap object
    vid.release()

    #Destroy all the windows
    cv2.destroyAllWindows()

#--------------------(Main)-------------------------#
openVideo()
