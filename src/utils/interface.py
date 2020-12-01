from argparse import ArgumentParser
import torchvision

def parse_cmd_args(arg=None):
    """ function used to parse the command line input of the autoencoder """

    # create the argument parser
    description = "Python script that creates a variational autoencoder used to reduce the dimensionality " \
                  "of the MNIST dataset."
    parser = ArgumentParser(description=description)

    # add an argument for the path of the dataset
    help = "The full/relative path to the file containing the training examples."
    parser.add_argument("-c", "--config", type=str, action="store", metavar="configuration_file_path",
                        required=True, help=help)

    # parse the arguments and return the result
    return parser.parse_args(arg)


def parse_config_file(filepath):
    """ function used to parse the configuration file given by the user """
    conv_layers = 3
    conv_channels = [16, 32, 64]
    conv_kernel_sizes = [(7, 7), (5, 5), (3, 3)]
    conv_strides = [(1, 1), (1, 1), (1, 1)]
    conv_paddings = [(1, 1), (1, 1), (1, 1)]

    beta = 0.5

    gamma = 1000
    capacity = 12.5

    configuration1 = {
        "variation": "VAE",
        "dataset": "MNIST",
    }

    configuration2 = {
        "variation": "B-VAE",
        "dataset": "MNIST",
        "beta": beta
    }

    configuration3 = {
        "variation": "C-VAE",
        "dataset": "MNIST",
        "gamma": gamma,
        "capacity": capacity
    }

    architecture = {
        "conv_layers": conv_layers,
        "conv_channels": conv_channels,
        "conv_kernel_sizes": conv_kernel_sizes,
        "conv_strides": conv_strides,
        "conv_paddings": conv_paddings
    }

    return (configuration1, architecture)


def load_dataset(dataset):
    ds_prompt = "\nEnter the path you want the dataset to be stored.\nIf you already have downloaded it, enter its path\n"
    if (dataset == "MNIST"):
        ds_path = input(ds_prompt)

        train_ds = torchvision.datasets.MNIST(root=ds_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
        test_ds = torchvision.datasets.MNIST(root=ds_path, train=False, download=True, transform=torchvision.transforms.ToTensor())

        return (train_ds, test_ds)
