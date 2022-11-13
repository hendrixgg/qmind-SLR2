#Import Lib
import os
import cv2
import numpy as np

#------------------(Functions)----------------------#

#greyscale images func


#


#--------------------(Main)-------------------------#

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

# Destroy all the windows
cv2.destroyAllWindows()