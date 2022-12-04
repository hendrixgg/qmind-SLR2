import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# get image data from files #
train_set, test_set = pd.read_csv('mnist_handsigns/sign_mnist_train.csv'), pd.read_csv('mnist_handsigns/sign_mnist_test.csv')

train_labels, train_images = train_set['label'].values, np.reshape(train_set.iloc[:, 1:].values, (27455, 28, 28)) / 255.0
test_labels, test_images = test_set['label'].values, np.reshape(test_set.iloc[:, 1:].values, (7172, 28, 28)) / 255.0

img = train_images[0]

# ROTATION #
rotated_img = ndimage.rotate(img, angle=45, reshape=False)

# BLUR #
blurred_img = ndimage.gaussian_filter(img, sigma=1)

# Showing images #
fig = plt.figure(figsize=(10, 3))
ax1, ax2, ax3 = fig.subplots(1, 3)

ax1.imshow(img, cmap='gray')
ax1.set_axis_off()
ax2.imshow(rotated_img, cmap='gray')
ax2.set_axis_off()
ax3.imshow(blurred_img, cmap='gray')
ax3.set_axis_off()
plt.show()