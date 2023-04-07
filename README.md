# QMIND x Distributive x JKUAT
## Project Bravo - Live ASL Recognition
## Table of Contents
1. Meet The Team
2. Required Libraries
3. Repository Organization
4. The Data Sets
5. The Model
6. Next Steps

### Meet The Team
- Hendrix Gryspeerdt - Project Lead
    - First year engineering student
    - hendrix.gryspeerdt@gmail.com
    - https://github.com/hendrixgg 
- Sammi Wang
    - 4th year Cognitive Science student
    - wxr.sammi@hotmail.com
    - https://github.com/SammiW15 
- Liam Salass
- Noah Waisbrod
    - https://github.com/Noah-Waisbrod 
- Matthew Li


### Required Libraries
to run locally:
- openCV
- MediaPipe
- numpy
- pandas
- process_image
- scipy
- matplotlib
- seaborn
- sklearn

to save and load model to onnx:
- skl2onnx
- onnxruntime

To create a conda environment copy and paste the following commands into the terminal one by one *****doesn't work***:

```
conda create -n qmind_slr python=3.10.6 tensorflow numpy pandas scipy scikit-learn matplotlib seaborn opencv pip
conda activate qmind_slr
pip install mediapipe
pip install skl2onnx
pip install onnxruntime
```

### Repository Organization
#### File List
- Demo.py
  - Main file to run for trying model. 
  - Imports WebCam.py
- WebCam.py
  - Python class for opening window and applying predictions
  - Imports live_predictor.py
- live_predictor.py
  - Used for sliding window prediction logic and printing command line text
- asl_model.py
  - Used for model evaluation and testing
- save_to_landmarks.py
  - Used to convert jpg hand images into landmark dataset
- process_image.py
  - Used to process live image into 
- landmarks_model.ipynb
  - Model training pipeline for SVM
- models/
  - Directory with saved models
- mnist_handsigns/
  - Stores data set #1


##### Branches
- main
  - most up to date branch with landmakrs2 model used in Demo.py
- landmarks2
  - Branch includes the pipeline used to train the final landmarks2 svm model
- landmarks
  - Initail landmarks pipeline and model branch
- model4
  - Final CNN model and pipeline using data set #3
- hand-detection
  - Branch used for working on Webcam.py, mediapipe interfacing and creating sliding window predictions
- model3
  - Branch with model3 trained on data set #2
- sav_img
  - Branch for working on Webcam.py, and saving/preprocessing images
- model2
  - Initial CNN pipeline and model trained on data set #1

### The Data sets

#### Data Set 1: Sign Language MNIST
https://www.kaggle.com/datasets/datamunge/sign-language-mnist 

The MNIST data set is composed of 27455 training and 7172 test data cases. Each training and test case represents a 28x28 pixel image (pixel1-784) with a label (0-25) corresponding to an alphabetic letter, excluding 9=J and 25=Z which required gesture motions. The data came from extending a small number of color images with various users and backgrounds doing hand signs. After the modifications, all the images were gray-scaled, cropped around the hand region of interest, and had low resolution.
The data set was available on Kaggle and was stored as csv files.

#### Data Set 2: Kaggle data set
https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-alphabets

The data set contains 27 classes: letters A-Z and unknown (images without hands in the frame). There was a total of 40500 images, 1500 for each class. It was not split into training and test data sets.
The images were grey-scaled, cropped to the region of interest, and resized to the same dimensions, but were squished and tilted so that the hands in the frame were not of the same size. The images mostly had white sheet backgrounds, but part of it had messy backgrounds which resembled the real applications. The data set is the smallest among the three, but had the highest variability among the data, though it still lacks a presentation of unique hands. 
The data set was available on Kaggle and was stored as jpg files.

#### Data Set 3: ASL Alphabet
https://www.kaggle.com/datasets/grassknoted/asl-alphabet 

The training data set contains 87,000 images in 200 x 200 pixels. It has 29 classes: 26 for letters A-Z (J and Z are presented without gestures) and 3 classes for SPACE, DELETE, and NOTHING. The NOTHIHG class represents a blank wall, which was equivalent to Mediapipe detecting no hand in the frame. These 3 classes allow transitions between spelling different letters in real-time applications and classifications. The test data set only contains 29 images, one for each class, which allows us to test the model against our live-stream data. 
The data in the data set are color images cropped around the hand region of interest. However, it is worth noting that the images all represent similar hands with similar backgrounds, providing little variety between the data. There is also an apparent lack of rotated hand signs. This resulted in letters such as 'L', 'D', and 'X' not being classifiable from their side views. 
The data set was available on Kaggle and GitHub and was stored as jpg files.

### The SVM Model (landmarks2)
#### Mediapipe library:
Mediapipe is a hand-detection open-source library produced by Google. It was utilized to convert a data set containing hand images into a new data set containing hand landmarks positions in 3 dimensions. There are 21 hand landmarks that media pipe would generate on a photo, however, only 20 points were used as inputs to the SVM. This was done by subtracting the position of the wrist landmark from all 20 other landmarks to get their relative positions to the wrist, as opposed to their relative position in the frame. The newly generated data set was then used as the training data for the SVM model. The library was also used to detect where the hand is in the frame and crop the hand's region to be passed to models (CNN models) as square images. That way, all images passed to the model avoid being compressed, and fit the models' input shape (in the case of the CNN). For the aforementioned reasons, Mediapipe was a necessary component to the success of our project. 

https://google.github.io/mediapipe/

#### The Support Vector Machine (SVM):
Support vector machines are effective at classifying sets of high-dimensional vectors. The intuition behind using a support vector machine to classify hand sign gestures is as follows. In the case of the hand landmarks generated by Mediapipe (see image above), the set of 3D points that form the hand orientation can be flattened into a vector representing a position in a high-dimensional space. For example, the space of hand landmark orientations that would be considered to be an A would form some region in the high dimensional space, and support vector machines are inherently effective at classifying these types of regions.
This model corresponds with our common way of understanding hand signs - according to the relative locations of the fingers, instead of the exact image of a hand. It is better to use one model to locate general hand features (Mediapipe), then another to parse those features into different hand signs (SVM).

### Next Steps
Our immediate next steps for improving our ASL prediction model involve increasing prediction accuracy and improving the control logic for creating strings of letters to make words. To achieve this, We will create our own dataset to ensure that it is more accurate and complete. The data sets our group could find all had inherent faults, ranging from too small data sets, and data sets with low variance in images, to having incorrect labeling for whole single letters. By including more images of hand signs taken from vertically rotated angles in the data set, the model will better predict letters that have been rotated. 

Additionally, we will add a feature to our ASL prediction model to identify numbers being held up and implement a model for identifying gestures and correlating them to phrases. We will begin by collecting data on the signed numbers and gestures commonly used to convey certain phrases to add these features to our model. We will then integrate this data into our model and train it to recognize these new features accurately. A rolling prediction would be necessary for creating a functioning model for gestures. 

Looking further ahead, we would like to develop an app that can perform these predictions, making ASL communication more accessible and efficient. With these improvements, we are confident that our ASL prediction model will become a valuable tool for promoting accessibility and inclusivity. 

  
