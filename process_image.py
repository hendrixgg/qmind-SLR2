import numpy as np
from scipy import ndimage
import cv2

#------------------(Functions)----------------------#

# testing ROTATION #
def rotate_image(img, angle):
    return ndimage.rotate(img, angle=angle, reshape=False, cval=1.0)

# testing BLUR #
def blur_image(img, blur_amount):
    return ndimage.gaussian_filter(img, sigma=blur_amount)

# Rescale image to grayscale numpy array (28, 28, 1) with values 0 to 1
# ASSUMING THE PIXEL VALUES ARE FROM 0-255 IN THE FIRST PLACE
def rescale_image(img):
    return np.asarray(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dsize=(28, 28), interpolation=cv2.INTER_CUBIC)) / 255.0

def rescale_image_from_file(img_path):
    return rescale_image(cv2.imread(img_path))


#--------------------(Main)-------------------------#
def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    # get image data from files #
    train_set, test_set = pd.read_csv('mnist_handsigns/sign_mnist_train.csv'), pd.read_csv('mnist_handsigns/sign_mnist_test.csv')

    train_labels, train_images = train_set['label'].values, np.reshape(train_set.iloc[:, 1:].values, (27455, 28, 28)) / 255.0
    test_labels, test_images = test_set['label'].values, np.reshape(test_set.iloc[:, 1:].values, (7172, 28, 28)) / 255.0

    img = train_images[0]
    raw_img = cv2.imread("LetterData/c/c0.png")
    print(raw_img)
    print(raw_img.shape)
    
    # Showing some modified images #
    fig = plt.figure(figsize=(10, 5))
    ax1, ax2, ax3, ax4, ax5 = fig.subplots(1, 5)

    ax1.imshow(img, cmap='gray')
    ax1.set_axis_off()
    ax2.imshow(rotate_image(img, 45), cmap='gray')
    ax2.set_axis_off()
    ax3.imshow(blur_image(img, 1), cmap='gray')
    ax3.set_axis_off()
    ax4.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY), cmap='gray')
    ax4.set_axis_off()
    print(rescale_image(raw_img))
    print(rescale_image(raw_img).shape)
    ax5.imshow(rescale_image(raw_img), cmap='gray')
    ax5.set_axis_off()
    plt.show()

if __name__ == "__main__":
    main()