#Import Lib
import os
import cv2
import live_predictor

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
    sign_language_model = live_predictor.live_asl_model()
    
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

        # frame -> cropped image, top predictions, text
        # modify the frame base on which overlays to display
        success, cropped, predictions, text = sign_language_model.process(frame, make_predictions=make_predictions, overlay_bounding_box=True, 
        overlay_landmarks=True, top_n=3)
        
        cv2.imwrite("temp_img.png", cropped) if success else 0

        # Display the resulting frame
        cv2.putText(frame, f"predicted letter: {predictions}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (209,80,0,255), 3)
        
        cv2.imshow('frame', frame)
        # cv2.displayOverlay('frame', f"predicted letter: {predictions}")
        # the overlay could include more information
        # the openVideo function could take in some more parameters, giving the option to show more data on overlay

        # print text to console
        os.system("cls")
        print(f"predicted letter: {predictions}")
        print("current text:\n", text)

        # hotkey assignment
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif capture_mode and timmer % scTime == 0:
            imgName = f"{imgCnt}.png"
            cv2.imwrite(os.path.join(path, imgName), cropped if success else frame)
            imgCnt += 1

        if cv2.waitKey(1) & 0xFF == ord('c'):
            sign_language_model.text_prediction.reset()
  
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