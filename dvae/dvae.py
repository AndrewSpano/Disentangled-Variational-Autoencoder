import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import logging
import sys
sys.path.append("../utils")
sys.path.append("dvae_utilities")

from utils import *
from dvae_module import *


def main(args):
    """ main() driver function """

    # first make sure that the path to the provided dataset is valid
    if filepath_is_not_valid(args.data):
        logging.error("The path {} is not a file. Aborting..".format(args.data))
        exit()

    # get the data from the training set
    X = parse_dataset(args.data)
    rows = X.shape[1]
    columns = X.shape[2]

    # reshape so that the shapes are (number_of_images, 1, rows, columns)
    X = X.reshape(-1, 1, rows, columns)
    # normalize
    X = X / 255.

    # split data to training and validation
    PANATHA = 13
    X_train, X_val = train_test_split(X, test_size=0.15, random_state=PANATHA, shuffle=True)

    # define here the hyperparameters that define the architecture of the model
    conv_layers = 3
    conv_kernel_sizes = [(7, 7), (7, 7), (7, 7)]
    pool_kernel_sizes = [(2, 2), (2, 2), (7, 7)]
    channels = [32, 64, 128]
    strides = [(1, 1), (1, 1), (1, 1)]

    # place them in a dictionary
    architecture_hyperparameters = {
        "conv_layers": conv_layers,
        "conv_kernel_sizes": conv_kernel_sizes,
        "pool_kernel_sizes": pool_kernel_sizes,
        "channels": channels,
    }

    # create the dvae
    dvae = DVAE(architecture_hyperparameters, latent_vector_dimension=2, beta=4)






if __name__ == "__main__":
    """ call main() function here """
    print()
    # configure the level of the logging and the format of the messages
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s\n")
    # parse the command line input
    args = parse_input(autoencoder=True)
    # call the main() driver function
    main(args)
    print("\n")
