import tensorflow as tf
import numpy as np
from process_image import rescale_image, rescale_image_from_file

model = tf.keras.models.load_model('models/ASL_model1')

#------------------(Functions)----------------------#

# prints the model architecture summary
def model_summary():
    model.summary()

# returns the letter label for a model output value
def get_label(integer_value):
    return chr(ord('A') + integer_value)

# Takes a (28, 28) numpy array with values 0-1 and returns the model output vector
# output has shape (, 25), one entry for the confidence of each letter's prediction
def predict(img):
    # there may be a better way to do this check. could take a look at the model.predict documentation.
    if not isinstance(img, np.ndarray) or img.shape != (28, 28):
        print("error in image prediction")
        return "ERROR"
    return model.predict(np.asarray([img]), verbose=0)[0]

# takes an unformatted cv2 image and predicts the ASL handsign in it
# ASSUMES THE IMAGE HAS PIXEL VALUES FROM 0-255
def predict_unformatted(img):
    return predict(rescale_image(img))

# takes a path to an image file (png or file readable by cv2)
def predict_file(img_path: str):
    return predict(rescale_image_from_file(img_path))

#--------------------(Main)-------------------------#
def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    # import test data
    test_set = pd.read_csv('mnist_handsigns/sign_mnist_test.csv')

    test_labels, test_imgs = test_set['label'].values, np.reshape(test_set.iloc[:, 1:].values, (len(test_set.index), 28, 28)) / 255.0

    # check the model
    test_loss, test_acc = model.evaluate(test_imgs,  test_labels, verbose=1)
    print("test_loss:", test_loss)
    print("test_acc:", test_acc)

    # test the model
    print("test_imgs[0]:")
    label = chr(ord('A') + test_labels[0])
    print(f"{label=} {predict(test_imgs[0])=}")

    print("raw image of a C:")
    label = 'C'
    raw_img = rescale_image_from_file("LetterData/c/c0.png")
    print(f"{label=} {predict(raw_img)=}")

    # showing the images

    fig = plt.figure(figsize=(10, 2))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.imshow(test_imgs[0], cmap="gray")
    ax1.set_axis_off()
    ax2.imshow(raw_img, cmap="gray")
    ax2.set_axis_off()
    plt.show()

if __name__ == "__main__":
    main()