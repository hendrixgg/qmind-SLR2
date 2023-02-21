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

- Convelutional 2D with rectified linear Unit activation function (ReLU)
- Batch Normalization layer (For re-centering and re-scalling)
- 2D Max Pool to downsize output
- 2nd Convelutional 2D

```
model = Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
```

### Next Steps

- Implement contol logic to chain together multiple live camera predictions to create words.
- Create a cropping method to isolate hands in video frame
- Test new models with transitioning gestures and implement J and Z
