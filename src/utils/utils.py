import struct
import numpy as np
import matplotlib.pyplot as plt


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
