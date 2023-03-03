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
        CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
                       "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]
         
        for category in CATEGORIES: 
            path = os.path.join(directory, category)   
            class_number = CATEGORIES.index(category) 
            print("Gathering jpg images of letter ", category)
            total_files = 0
            count = 0
            # Iterate directory
            for path_list in os.listdir(path):
                # check if current path is a file
                if os.path.isfile(os.path.join(path, path_list)):
                    total_files += 1
            print('File count:', total_files)
            
            for img in os.listdir(path):
                percent = round((count / total_files) * 100)
                count += 1
                print (percent, "%", end="\r")
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

        return self.data



    def split_data(self):
        img_size = 64
        x = []
        y = []
        for features, label in self.data:
            x.append(features)
            y.append(label)

        self.train_set = np.array(x).reshape(-1,img_size,img_size,1)
        print (self.train_set[0])

        self.train_labels = LabelBinarizer().fit_transform(np.array(y))

    def pre_process_data(self):
        self.train_set = self.train_set/255.0


        self.datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 20)
                    zoom_range = 0.1, # Randomly zoom image 
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=True)  # randomly flip images
        self.datagen.fit(self.train_set)
       
    def train_model(self):
            #Training the model
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

            model = Sequential()
            model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape=self.train_set.shape[1:]))
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
            model.add(Dense(units = 29 , activation = 'softmax'))
            model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
            model.summary()

            model.fit(self.train_set, self.train_labels, batch_size = 64, epochs = 4, validation_split=0.1, callbacks = [learning_rate_reduction])
            
            # Save the entire model as a SavedModel.
            model.save('models/asl_model4')

def main():

    #For dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    model = cnn()
    data = model.add_data('asl_alphabet')
    model.split_data()
    model.pre_process_data()
    model.train_model()



if __name__ == "__main__":
    import tensorflow as tf
    print("# GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()