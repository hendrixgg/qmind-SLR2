#Import Statements
import os, sys
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



class cnn_tf:
    #Initialize pipeline
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.train_labels = None
        self.test_labels = None
        self.datagen = None

    #Function for loading in dataframes
    def load(self, type, path):
        df = pd.read_csv(path)
        df_labels = df['label']
        df.drop(columns = 'label', inplace = True)
        if type == 'train':
            self.train_set = df
            self.train_labels = df_labels
            print ('Loaded training set')
        if type == 'test':
            self.test_set = df
            self.test_labels = df_labels
            print('Loaded testing set')

    #First preprocessing pipeline using https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy 
    def pre_process1(self):
        try:
            #Transfor multi-class labels to binary labels
            from sklearn.preprocessing import LabelBinarizer
            label_bin = LabelBinarizer()
            self.train_labels = label_bin.fit_transform(self.train_labels)
            self.test_labels = label_bin.fit_transform(self.test_labels)

            self.train_set = self.train_set.values    
            self.test_set = self.test_set.values 

            self.train_set = self.train_set/255
            self.test_set = self.test_set/255

            #Reshape 1-D to 3-D
            self.train_set = self.train_set.reshape(-1,28,28,1)
            self.test_set = self.test_set.reshape(-1,28,28,1)

            #Rotate, blur, shift, zoom images
            self.datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=50,  # randomly rotate images in the range (degrees, 0 to 180)
                    zoom_range = 0.4, # Randomly zoom image 
                    width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=True)  # randomly flip images
            self.datagen.fit(self.train_set)
        except:
            print ('Error in preprocessing: Make sure two dataframes are entered before')


    def pre_process2(self):
        try:
           print ('Not finished')
        except:
            print ('Error in preprocessing2: Make sure two dataframes are entered before')

    #First cnn model and training using https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy 
    def cnn_model1(self):
        try:
            #Training the model
            learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

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
            model.summary()

            history = model.fit(self.datagen.flow(self.train_set,self.train_labels, batch_size = 128) ,epochs = 20 , validation_data = (self.test_set, self.test_labels) , callbacks = [learning_rate_reduction])

            print("Accuracy of the model is - " , model.evaluate(self.test_set,self.test_labels)[1]*100 , "%")
            
            # Save the entire model as a SavedModel.
            model.save('models/asl_model2.1')
            
        except:
            print ('Error in cnn_model1')
            sys.exit(1)


    def cnn_model2(self):
        try:
            print ('Not finished')
        except:
            print('Error in cnn_model2')
            sys.exit(1)


def main():
    #Used to check cmd arguments
    # argv[1] holds an int for determining which preprocessing method to use
    # argv[2] hold an int for determining which cnn to use
    if len(sys.argv) < 3 or sys.argv[1] > 4 or sys.argv[1] < 1 or sys.argv[2] > 4 or sys.argv[2] < 1:
        #If nothing in cmd line or argv[] values outisde of range, go to default
        print('Usage: %s preprocessing_pipeline(1,2,3) cnn_pipeline(1,2,3)' % sys.argv[0])
        print('Using preprocessing: 1 CNN: 1')
        prep = 1
        cnn = 1
    else:
        #Set which pipelines to use
        prep = sys.argv[1]
        cnn = sys.argv[2]
        print('Using preprocessing: ' + prep + ' CNN: ' + cnn)

    

    #Initialize class and load data
    model = cnn_tf()
    model.load('train','mnist_handsigns/sign_mnist_train.csv')
    model.load('test','mnist_handsigns/sign_mnist_test.csv')
    
    #Pipeline 1 is using https://www.kaggle.com/code/madz2000/cnn-using-keras-100-accuracy 
    if prep == 1:
        model.pre_process1()
    if cnn == 1:
        model.cnn_model1()
    
    #Pipeline 2 is by Liams
    if prep == 2:
        model.pre_process2()
    if cnn == 2:
        model.cnn_model2()

    
#Call to main function
if __name__ == "__main__":
    main()