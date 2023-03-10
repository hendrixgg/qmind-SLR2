import asl_model
from process_image import rescale_image_from_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# x must be a one dimensional np.ndarray of integer values in the interval [0, length)
# returns a new array v with v.shape = (x.shape[0], length) and
# for any i, j, 0 <= i < x.shape[0], 0 <= j < length: v[x[i]] = 1 and v[j] = 0
# this is similar to scikitlearn's LabelBinarizer
def make_binary_vector(x: np.ndarray, length: int):
    v = np.zeros((x.shape[0], length))
    for xi, vi in zip(x, v):
        vi[xi] = 1.
    return v

shift = lambda x: np.where(x > 9, x - 1, x)

#--------------------(Main)-------------------------#
def main():
    # import test data
    test_set = pd.read_csv('mnist_handsigns/sign_mnist_test.csv')

    test_labels, test_imgs = make_binary_vector(shift(test_set['label'].values), length=24), np.reshape(test_set.iloc[:, 1:].values, (len(test_set.index), 28, 28)) / 255.0

    # check the model
    model = asl_model.Model("models/asl_model2")
    test_loss, test_acc = model.model.evaluate(test_imgs,  test_labels, verbose=1)
    print("test_loss:", test_loss)
    print("test_acc:", test_acc)

    # test the model
    ## img1
    print("test_imgs[0]:")
    label = asl_model.get_label(np.argmax(test_labels[0]))
    output = asl_model.predict(test_imgs[0])
    prediction = np.argmax(output)
    print(f"{label=} {output=}")
    print(f"model thinks that this is a: {asl_model.get_label(prediction)}")
    print(f"confidence: {output[prediction]}")

    ## img2
    print("raw image of a C:")
    label = 'C'
    raw_img = rescale_image_from_file("LetterData/c/c0.png")
    output = asl_model.predict(raw_img)
    prediction = np.argmax(output)
    print(f"{label=} {output=}")
    print(f"model thinks that this is a: {asl_model.get_label(prediction)}")
    print(f"confidence: {output[prediction]}")
    
    ## img3
    print("not an image of any letter")
    label = "NA"
    raw_img2 = rescale_image_from_file("letterData/NA/0.png")
    output = asl_model.predict(raw_img2)
    prediction = np.argmax(output)
    print(f"{label=} {output=}")
    print(f"model thinks that this is a: {asl_model.get_label(prediction)}")
    print(f"confidence: {output[prediction]}")

    # showing the images

    fig = plt.figure(figsize=(10, 3))
    ax1, ax2, ax3 = fig.subplots(1, 3)

    ax1.imshow(test_imgs[0], cmap="gray")
    ax1.set_axis_off()
    ax2.imshow(raw_img, cmap="gray")
    ax2.set_axis_off()
    ax3.imshow(raw_img2, cmap="gray")
    ax3.set_axis_off()
    plt.show()

if __name__ == "__main__":
    main()