#Import Statements
import os, sys
import random
from itertools import chain
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
try:
    import pandas as pd
except:
    print( 'Error: pandas has not been installed.' )
    sys.exit(0)
try:
    import mediapipe as mp
except:
    print( 'Error: mediapipe has not been installed' )
    sys.exit(0)

mp_hands = mp.solutions.hands


# saves an image dataset in the form of the hand landmarks produced my mediapipe.solutions.Hands()
class manage_dataset():

    def __init__(self, img_size=200):
        self.img_size = img_size
        self.data = []
        self.images = []
        self.labels = []
        self.landmark_rows = []
        self.datagen = None
        self.dataframe = pd.DataFrame()
        # mediapipe resource
        self.hands = mp_hands.Hands(max_num_hands=1, static_image_mode=True)

        # returns the hand landmarks array generated from a file
    def produce_hand_landmarks(self, img):
        raw_hand_landmarks = self.hands.process(img).multi_hand_landmarks
        if not raw_hand_landmarks:
            return False, False
        for hand_landmarks in raw_hand_landmarks:
            return True, hand_landmarks.landmark

    def add_images(self, directory):
        #Add data function is passed parent directory to directories with the same names as the catagories list below. 
        # It will add all images to the dataset that's directories match
        
        CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del']
        hand_found = []
        for category in CATEGORIES: 
            path = os.path.join(directory, category)
            class_number = CATEGORIES.index(category)
            img_count, no_hand = 0, 0

            print("Gathering jpg images of letter ", category)
            for img_name in os.listdir(path):
                img_count += 1
                img = cv2.cvtColor(cv2.imread(os.path.join(path, img_name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                success, landmarks = self.produce_hand_landmarks(img)
                if not success:
                    # print(f"did not find hand in {img_name}")
                    no_hand += 1
                    continue
                self.data.append((landmarks, class_number))
            hand_found.append(f"{img_count - no_hand}/{img_count} images of letter {category}")
            print(hand_found[-1])
        
        # print how many hands were found in images of certain categories
        print("images collected:")
        for found in hand_found:
            print(found)
        
        random.shuffle(self.data)
        # Check if the data was shuffled
        print('Checking if data is randomized')
        for sample in self.data[:10]:
            print(sample[1])

    # take the data store in self.data and split it into two numpy arrays, one containing landmarks and one being the label of the hand sign
    def split_data_landmarks(self):
        self.landmark_rows = []
        self.labels = []
        for landmarks, label in self.data:
            # wrist point
            wr = landmarks[0]
            # all of the points coordinates in a one dimesional array, with the wrist point at the origin
            # scale the width since points are relative to frame size (assume height of image to have length 1)
            in_line_landmarks = [*chain(*chain([p.x - wr.x, p.y - wr.y, p.z - wr.z] for p in landmarks))]
            self.landmark_rows.append(in_line_landmarks)
            self.labels.append(label)
        self.landmark_rows = np.array(self.landmark_rows).reshape(-1, len(self.landmark_rows[0]))
        self.labels = np.array(self.labels)

    def split_data_images(self):
        self.images = []
        self.labels = []
        for img, label in self.data:
            
            self.images.append(img)
            self.labels.append(label)
        # convert numpy arrays
        self.images = np.array(self.images).reshape((-1, self.img_size, self.img_size, 3))
        self.labels = np.array(self.labels)

    # consider flipping images
    def augment_images(self):
        self.datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 20)
                    zoom_range = 0.1, # Randomly zoom image 
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=False)  # randomly flip images
        self.datagen.fit(self.images)

    # this function will take the images saved in self.images and 
    def convert_to_landmarks(self):
        for img in self.images:
            success, landmarks = self.produce_hand_landmarks(img)
            if not success:
                print(f"no hand in image: {img}")
                continue
            # wrist point
            wr = landmarks[0]
            # all of the points coordinates in a one dimesional array, with the wrist point at the origin
            # scale the width since points are relative to frame size (assume height of image to have length 1)
            scale_factor = img.shape[1] * 1. / img.shape[0]
            in_line = [*chain(*chain([(p.x - wr.x) * scale_factor, p.y - wr.y, p.z - wr.z] for p in landmarks))]
            self.landmark_rows.append(in_line)


    def save_landmarks_to_csv(self, directory):
        # add data to dataframe
        self.dataframe = pd.DataFrame(self.landmark_rows, columns=[*chain(*chain([f"x{i}", f"y{i}", f"z{i}"] for i in range(21)))])
        self.dataframe["label"] = self.labels
        # write to .csv file
        self.dataframe.to_csv(os.path.join(directory, "asl_alphabet_landmarks.csv"))

# print([*chain(*chain([f"x{i}", f"y{i}", f"z{i}"] for i in range(21)))])

def main():
    data_converter = manage_dataset(img_size=200)
    data_converter.add_images("asl_alphabet_train\\asl_alphabet_train")
    # data_converter.augment_images()
    # data_converter.convert_to_landmarks()
    data_converter.split_data_landmarks()
    data_converter.save_landmarks_to_csv("asl_alphabet_train")


if __name__ == "__main__":
    main()