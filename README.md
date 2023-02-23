# QMIND x Distributive x JKUAT
## Project Bravo - Live ASL Recognition

### Meet The Team
- Matthew Li
- Sammi Wang
    - 4th year Cognitive Science student
- Liam Salass
- Noah Waisbrod
- Hendrix Gryspeerdt
    - First year engineering student

### Required Libraries

- openCV
- MediaPipe
- numpy
- pandas
- process_image
- scipy
- matplotlib
- seaborn
- TensorFlow
- 

To create a conda environment copy and paste the following commands into the terminal one by one:

```
conda create -n qmind_slr python=3.8.8 tensorflow numpy pandas scipy matplotlib seaborn pip opencv
conda activate qmind_slr
pip install mediapipe
```

### The Dataset

The dataset used was the Mnist Signlanguage Dataset from Kaggle:
https://www.kaggle.com/datasets/datamunge/sign-language-mnist

This dataset includes all alphabetical letters except 'J' and 'Z' due to said letter requiring complex gestures to represent and not static hand positions. 

The shape of the dataset is as follows:
27,455 cases in train set, 7,172 cases in test set, labels 0 - 25 to predict (J = 9 and Z = 25 are empty ) and 28 x 28 pixels with greyscale values between 0 - 255.

Datageneration was applied to the training set, resulting in added cases with blurred, noisy, and rotated images. 

### The Model
- Multiple models were made using Tensorflow2 Keras api. The main model being used is model2
- The model is located in /models/asl_model2
- The model had returned an accuracy of 99.74%. 
- A learning rate reduction was used to slow the learning process and prevent overcorrections.
- The model was sequential due to the simplistic predictions being made. 

#### Model Structure:
Learning rate reduction schedule using reduce on plateu in order dynamically adjust learning rate during training when model prediction accuracy has minimal change between epochs. 

1.  Convelutional 2D with rectified linear Unit activation function (ReLU)
2.  Batch Normalization layer (For re-centering and re-scalling)
3.  2D Max Pool to downsize output
4.  2nd Convelutional 2D with ReLU activation function 
5.  Dropout layer to avoid overfitting and weaken neurons
6.  2nd Batch Normalization layer
7.  2nd 2D Max Pool 
8.  3rd Convelutional 2D with ReLU
9.  3rd Batch Normalization Layer
10. 3rd 2D Max Pool
11. Flatten layer to create linear vector output of data
12. Dense layer to classify output of convelutional layers (ReLU activation function)
13. 2nd Droupout layer
14. 2nd Dense layer (softmax activation function)
15. Adam compiler
 

### Next Steps
#### Short term:
- Liam:
  - Retrain model for square hands 
  - Look into finding handsign data set with hand lines drawn
  - Higher resolution dataset with dataset 
  - Increase model accuracy across the board
  - Get Demo.py to work
- Sammi:
  - Fix webcam text display to work (either create new window for predictions or add area below window)
  - Fix Sammi's ability to push/pull with repository
  - Add no hand detected mode 
  - Poster Board for Cucai
- Hendrix:
  - Cropped images to be square
  - Stringing together words from predictions

#### Long Term:

  
