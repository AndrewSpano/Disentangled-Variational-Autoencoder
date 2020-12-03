import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def filepath_is_not_valid(filepath):
    """ function used to check whether a filepath containing information is valid """

    # check if the path leads to a file
    if not os.path.isfile(filepath):
        # return false
        return True

    # return false since the path is valid
    return False

def prepare_dataset(configuration):
    dataset_info = {}
    if (configuration["dataset"] == "MNIST"):
        dataset_info["ds_method"] = torchvision.datasets.MNIST
        dataset_info["ds_shape"] = (1, 28, 28)
        dataset_info["ds_path"] = configuration["path"]
    elif (configuration["dataset"] == "CIFAR10"):
        dataset_info["ds_method"] = torchvision.datasets.CIFAR10
        dataset_info["ds_shape"] = (3, 32, 32)
        dataset_info["ds_path"] = configuration["path"]
    else:
        print("Currently only MNIST & CIFAR10 datasets are supported")
        return None

    return dataset_info

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

def str_to_int_list(string):
    list = []
    parts = string.split(',')

    for part in parts:
        part = part.replace('[', '')
        part = part.replace(']', '')
        part = part.strip()

        number = int(part)
        list.append(number)

    return list

def str_to_tuple_list(string):
    list = []
    parts = string.split(')')

    for part in parts:
        part = part.replace('[', '')
        part = part.replace(']', '')
        part = part.replace('(', '')
        part = part.strip()

        inner_parts = part.split(',')

        inner_list = []
        for inner_part in inner_parts:
            if (inner_part == ''):
                continue
            inner_part = inner_part.strip()
            number = int(inner_part)
            inner_list.append(number)

        inner_tuple = tuple(inner_list)
        if (len(inner_tuple) == 2):
            list.append(inner_tuple)

    return list

def plot_image(image):
    """ fuction used to plot an image using matplotlib """

    # plot and show the image
    plt.imshow(image, cmap="gray")
    plt.show()

def plot_against(image1, image2, label, cmap):
    fig=plt.figure(figsize=(6, 6))
    title = "Label {}".format(label)
    fig.suptitle(title, fontsize=12)
    columns = 2
    rows = 1

    im1 = fig.add_subplot(rows, columns, 1)
    title1 = "Original"
    im1.title.set_text(title1)
    plt.imshow(image1, cmap=cmap)

    im2 = fig.add_subplot(rows, columns, 2)
    title2 = "Generated"
    im2.title.set_text(title2)
    plt.imshow(image2, cmap=cmap)


    plt.show()
