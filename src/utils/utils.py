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
    """ function used to set some values used by the model based on the dataset selected """
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

def str_to_int_list(string):
    """ utility function used to convert a string to a list of integers """
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
    """ utility function used to convert a string to a list of tuples of integers """
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

def plot_against(image1, image2, label, cmap):
    """ function used to plot 2 images "against" each other using pyplot """
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
