# For saving images as landmarks
- take in all images and labels
- modify some images to generate some modified data
- create a pandas dataframe with:
    - one row per image
    - 63 columns for the 21 (x, y, z) (float value)
    - one more column for the label (0-26 integer)
- iterate over image-label pair
    - generate landmarks with mediapipe
    - if there is no hand landmarks generated, make all landmarks to be (0, 0, 0), skip this image
    - save to row in dataframe
- save dataframe to .csv file

# reading csv data to train model
- read the .csv file into a pandas dataframe and save data into two numpy arrays, one with landmarks, one with labels
    - could augment datapoints, but something to try later
- build model architecture
- train model on dataset
- test with real images

# For live predicting using hand landmark model
- process image with mediapipe hands to get landmarks
- pass in landmarks into formatting function located in ai model python module
    - depends on the model architecture, but probably just a single dimensional array as input
- run prediction based on landmarks
- return prediction to the WebCam.py
- continue with control logic as before