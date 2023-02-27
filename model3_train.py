#Import Statements
import os, sys
import random
try:
    import numpy as np
except:
    print( 'Error: numpy has not been installed.' )
    sys.exit(0)
try:
    import pandas as pd
except:
    print( 'Error: pandas has not been installed.' )
    sys.exit(0)
try:
    import matplotlib.pyplot as plt
except:
    print( 'Error: matplotlib has not been installed.' )
    sys.exit(0)
try:
    import seaborn as sns
except:
    print( 'Error: seaborn has not been installed.' )
    sys.exit(0)
try:
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ReduceLROnPlateau
except:
    print( 'Error: tensorflow has not been installed.' )
    sys.exit(0)
try:
    from sklearn.preprocessing import LabelBinarizer
except:
    print('Error : scikit-learn has not been installed')
try:
    import cv2
except:
    print( 'Error: opencv has not been installed.' )
    sys.exit(0)


class cnn:

    def __init__(self):
        #Initialize datasets
        self.data = []
        self.train_set = None
        self.train_labels = None
        self.datagen = None
 

    def add_data(self, directory):
        #Add data function is passed parent directory to directories with the same names as the catagories list below. 
        # It will add all images to the dataset that's directories match

        img_size = 64
        CATEGORIES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
              "u", "v", "w", "x", "y", "z", "unknown"]
        for category in CATEGORIES: 
            path = os.path.join(directory, category)   
            class_number = CATEGORIES.index(category) 
            print("Gathering jpg images of letter ", category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    img_array_resized = cv2.resize(img_array, (img_size, img_size))
                    self.data.append([img_array_resized, class_number])
                except Exception as e:
                    pass
        random.shuffle(self.data)
        #Check if shuffled
        print('Checking if data is randomized')
        for sample in self.data[:10]:
            print(sample[1])



    def split_data(self):
        img_size = 64

        for features,label in self.data:
            self.train_set = (features)
            self.train_labels = (label)

        self.train_set = np.array(self.train_set).reshape(-1,img_size,img_size,1)
        print (self.train_set[0])

    def pre_process_data(self):
        self.train_set = self.train_set/255.0


        self.datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                    zoom_range = 0.1, # Randomly zoom image 
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=True)  # randomly flip images
        self.datagen.fit(self.train_set)
       
    def train_model(self):
            img_size = 64
            #Training the model
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

            model = Sequential()
            model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape =self.train_set.shape[1:]))
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
            model.add(Dense(units = 27 , activation = 'softmax'))
            model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
            model.summary()

            history = model.fit(self.datagen.flow(self.train_set,self.train_labels, batch_size = 128) ,epochs = 20 , validation_split = 0.1 , callbacks = [learning_rate_reduction])
            
            # Save the entire model as a SavedModel.
            model.save('models/asl_model3')

def main():
    #For dataset https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-alphabets?resource=download 
    model = cnn()
    model.add_data('handsign_imgs')
    model.split_data()
    model.pre_process_data()
    model.train_model()



if __name__ == "__main__":
    main()