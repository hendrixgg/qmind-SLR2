#Import Lib
import os
import cv2
import asl_model
import live_predictor
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
    #open webcam and show in folder
    vid = cv2.VideoCapture(0)
    imgCnt = 0
    timmer = 0
    while(True):
        timmer += 1
        # Capture the video frame
        ret, frame = vid.read()

        # predict the current letter
        rolling_prediction.add_vector(asl_model.predict_unformatted(frame))
        # get the top 3 predictions
        predictions = [(asl_model.get_label(i), c) for (i, c) in rolling_prediction.get_confidences(3)]
        # if there has been low confidence in gestures, treat the situation as a "transition between gestures" and perhaps add the previouly predicted letter to a string to record the interpretations
        # if a "transition" was the previous "letter" do not add anything to the string recording the interpretation
        # to make the framerate not be so slow, could find a way to make this asyncronus

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.displayOverlay('frame', f"predicted letter: {predictions}")
        # the overlay could include more information
        # the openVideo function could take in some more parameters, giving the option to show more data on overlay

        # hotkey assignment
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif capture_mode and timmer % scTime == 0:
            imgName = f"{imgCnt}.png"
            cv2.imwrite(os.path.join(path, imgName),frame)
            imgCnt += 1
  
    # After the loop release the cap object
    vid.release()

    #Destroy all the windows
    cv2.destroyAllWindows()

#--------------------(Main)-------------------------#
def main():    
    dir = createDir('c')
    openVideo(dir, 150, make_predictions=False)

if __name__ == "__main__":
    main()
