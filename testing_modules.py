import asl_model
from process_image import rescale_image_from_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#--------------------(Main)-------------------------#
def main():
    # import test data
    test_set = pd.read_csv('mnist_handsigns/sign_mnist_test.csv')

    test_labels, test_imgs = test_set['label'].values, np.reshape(test_set.iloc[:, 1:].values, (len(test_set.index), 28, 28)) / 255.0

    # check the model
    # test_loss, test_acc = asl_model.model.evaluate(test_imgs,  test_labels, verbose=1)
    # print("test_loss:", test_loss)
    # print("test_acc:", test_acc)

    # test the model
    ## img1
    print("test_imgs[0]:")
    label = asl_model.get_label(test_labels[0])
    print(f"{label=} {asl_model.predict(test_imgs[0])=}")
    print(f"model thinks that this is a: {asl_model.get_label(np.argmax(asl_model.predict(test_imgs[0])))}")

    ## img2
    print("raw image of a C:")
    label = 'C'
    raw_img = rescale_image_from_file("LetterData/c/c0.png")
    print(f"{label=} {asl_model.predict(raw_img)=}")
    print(f"model thinks that this is a: {asl_model.get_label(np.argmax(asl_model.predict(raw_img)))}")
    
    ## img3
    print("not an image of any letter")
    label = "NA"
    raw_img2 = rescale_image_from_file("letterData/NA/0.png")
    print(f"{label=} {asl_model.predict(raw_img2)=}")
    print(f"model thinks that this is a: {asl_model.get_label(np.argmax(asl_model.predict(raw_img2)))}")

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