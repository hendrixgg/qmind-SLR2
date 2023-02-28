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
def rescale_image(img, shape=(28, 28)):
    return np.asarray(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dsize=shape, interpolation=cv2.INTER_CUBIC)) / 255.0

# Reads the image from the specified filepath, returns it after applying rescale_image
def rescale_image_from_file(img_path, shape=(28, 28)):
    return rescale_image(cv2.imread(img_path), shape=shape)

# cropping an image given a list of points with attributes .x, and .y
# if cropp is outside of input image, fill out of bound pixels with fill_pixel
def crop_square(img, points, pad_left=0, pad_right=0, pad_top=0, pad_bottom=0, fill_pixel=[255, 255, 255]):
    top, bottom = (img.shape[1], img.shape[0]), (0, 0)
    # find extreme points
    for pt in points:
        x, y = int(pt.x * img.shape[1]), int(pt.y * img.shape[0])
        top = (min(top[0], x), min(top[1], y))
        bottom = (max(bottom[0], x), max(bottom[1], y))

    # pad into a square
    w, h = bottom[0] - top[0], bottom[1] - top[1]
    if w < h:
        pad_left += (h - w) // 2
        pad_right += (h - w) // 2
    elif h < w:
        pad_top += (w - h) // 2
        pad_bottom += (w - h) // 2
    # add padding
    top = (top[0] - pad_left, top[1] - pad_top)
    bottom = (bottom[0] + pad_right, bottom[1] + pad_bottom)
    # create the empty cropped image
    cropped = np.full((bottom[1] - top[1] + 1, bottom[0] - top[0] + 1, 3), np.asarray(fill_pixel, dtype=np.uint8))
    # put in pixel values from input image
    x_min, y_min, x_max, y_max = max(top[0], 1), max(top[1], 1), min(bottom[0], img.shape[1] - 1), min(bottom[1], img.shape[0] - 1)
    cropped[y_min - top[1]:y_max - top[1] + 1, x_min - top[0]:x_max - top[0] + 1] = img[y_min:y_max + 1, x_min:x_max + 1]

    return cropped, top, bottom
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