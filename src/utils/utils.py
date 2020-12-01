import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def filepath_is_not_valid(filepath):
    """ function used to check whether a filepath containing information is valid """

    # check if the path leads to a file
    if not os.path.isfile(filepath):
        # return false
        return True

    # return false since the path is valid
    return False

def parse_dataset(filepath):
    """ function used to parse the data of a dataset """

    # open the dataset
    with open(filepath, "rb") as dataset:
        # read the magic number and the number of images
        magic_number, number_of_images = struct.unpack(">II", dataset.read(8))
        # read the number of rows and number of columns per image
        rows, columns = struct.unpack(">II", dataset.read(8))
        # now read the rest of the file using numpy.fromfile()
        images = np.fromfile(dataset, dtype=np.dtype(np.uint8).newbyteorder(">"))
        # reshape so that the final shape is (number_of_images, rows, columns)
        images = images.reshape((number_of_images, rows, columns))

    # return the images
    return images


def parse_labelset(filepath):
    """ function used to parse the data of a labelset """

    # open the file
    with open(filepath, "rb") as labelset:
        # read the magic number and the number of labels
        magic_number, number_of_labels = struct.unpack(">II", labelset.read(8))
        # now read the rest of the file using numpy.fromfile()
        labels = np.fromfile(labelset, dtype=np.dtype(np.uint8).newbyteorder(">"))

    # return the labels
    return labels


def plot_image(image):
    """ fuction used to plot an image using matplotlib """

    # plot and show the image
    plt.imshow(image, cmap="gray")
    plt.show()

def plot_against(image1, image2):
    fig=plt.figure(figsize=(6, 6))
    columns = 2
    rows = 1

    im1 = fig.add_subplot(rows, columns, 1)
    title1 = "Original"
    im1.title.set_text(title1)
    plt.imshow(image1, cmap='gray')

    im2 = fig.add_subplot(rows, columns, 2)
    title2 = "Generated"
    im2.title.set_text(title2)
    plt.imshow(image2, cmap='gray')


    plt.show()
