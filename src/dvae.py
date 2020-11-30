import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import logging
import sys
sys.path.append("./models")
sys.path.append("./utils")

from utils import *
from vae import VAE


def main():
    """ main() driver function """

    path = "../Dataset/train-images-idx3-ubyte"

    # get the data from the training set
    X = parse_dataset(path)
    rows = X.shape[1]
    columns = X.shape[2]

    # reshape so that the shapes are (number_of_images, 1, rows, columns)
    X = X.reshape(-1, 1, rows, columns)
    # normalize
    X = X / 255.

    X = torch.from_numpy(X)

    # split data to training and validation
    PANATHA = 13
    X_train, X_val = train_test_split(X, test_size=0.15, random_state=PANATHA, shuffle=True)

    # define here the hyperparameters that define the architecture of the model
    conv_layers = 3
    conv_channels = [16, 32, 64]
    conv_kernel_sizes = [(7, 7), (5, 5), (3, 3)]
    conv_strides = [(1, 1), (1, 1), (1, 1)]
    conv_paddings = [(1, 1), (1, 1), (1, 1)]

    # place them in a dictionary
    architecture_hyperparameters = {
        "conv_layers": conv_layers,
        "conv_channels": conv_channels,
        "conv_kernel_sizes": conv_kernel_sizes,
        "conv_strides": conv_strides,
        "conv_paddings": conv_paddings
    }

    # plot_image(X[0, 0, :, :])
    # create the dvae
    vae = VAE(architecture_hyperparameters, input_shape=X.shape, z_dimension=2)



    output, mean, std = vae(X[0:1, :, :, :].float())
    # output.detach().numpy()

    # plot_image(output.detach().numpy()[0, 0, :, :])

    x_for_loss = X[0:1, :, :, :]
    output_for_loss = output[0:1, :, :, :]
    loss = vae.criterion(x_for_loss, output_for_loss, mean, std)

    print("loss = {}".format(loss))


if __name__ == "__main__":
    """ call main() function here """
    print()

    # call the main() driver function
    main()
    print("\n")
