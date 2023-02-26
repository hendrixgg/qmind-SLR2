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

try:
    import cv2
except:
    print( 'Error: opencv has not been installed.' )
    sys.exit(0)


class cnn:

    def __init__(self):
        #Initialize datasets
        self.train_set = None
        self.test_set = None
        self.train_labels = None
        self.test_labels = None
        self.datagen = None

    def get_files(directory):
        files = []
        for dirname, _, filenames in os.walk(directory):
            for filename in filenames:
                print(filename, " added to files")
                files.append(os.path.join(dirname, filename))
        return files
    

def main():
    model = cnn()
    test = model.get_files('/handsign_imgs/a/')



if __name__ == "__main__":
    main()