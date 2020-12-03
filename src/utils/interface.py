from argparse import ArgumentParser
from configparser import ConfigParser
import torchvision

from utils import str_to_int_list, str_to_tuple_list

def parse_cmd_args(arg=None):
    """ function used to parse the command line input of the autoencoder """

    # create the argument parser
    description = "Python script that creates a variational autoencoder used to reduce the dimensionality " \
                  "of the MNIST dataset."
    parser = ArgumentParser(description=description)

    # add an argument for the path of the dataset
    help = "The full/relative path to the file containing the config file."
    parser.add_argument("-c", "--config", type=str, action="store", metavar="configuration_file_path",
                        required=True, help=help)


    help = "The variation of VAE to be trained (VAE, beta-VAE)"
    parser.add_argument("-v", "--variation", type=str, action="store", metavar="model_variation",
                        required=True, help=help)

    # parse the arguments and return the result
    return parser.parse_args(arg)


def parse_config_file(filepath, variation):
    """ function used to parse the configuration file given by the user """

    configuration = {}
    architecture = {}
    hyperparameters = {}

    available_config_params = ['dataset', 'path']

    available_arch_params = ['conv_layers', 'conv_channels', 'conv_kernel_sizes',
                            'conv_strides', 'conv_paddings', 'z_dimension']

    available_hyper_params = ['epochs', 'batch_size', 'beta']

    config = ConfigParser()
    config.read(filepath)

    dataset = config.get('configuration', 'dataset')
    path = config.get('configuration', 'path')

    conv_layers_str = config.get('architecture', 'conv_layers')
    conv_layers = int(conv_layers_str)

    conv_channels_str = config.get('architecture', 'conv_channels')
    conv_channels = str_to_int_list(conv_channels_str)
    if (len(conv_channels) != conv_layers):
        print("The amount of convolution channels provided needs to be equal to the amount of convolutional layers")

    conv_kernel_sizes_str = config.get('architecture', 'conv_kernel_sizes')
    conv_kernel_sizes = str_to_tuple_list(conv_kernel_sizes_str)
    if (len(conv_kernel_sizes) != conv_layers):
        print("The amount of convolution channels provided needs to be equal to the amount of convolutional layers")

    conv_strides_str = config.get('architecture', 'conv_strides')
    conv_strides = str_to_tuple_list(conv_strides_str)
    if (len(conv_strides) != conv_layers):
        print("The amount of convolution channels provided needs to be equal to the amount of convolutional layers")

    conv_paddings_str = config.get('architecture', 'conv_paddings')
    conv_paddings = str_to_tuple_list(conv_paddings_str)
    if (len(conv_paddings) != conv_layers):
        print("The amount of convolution channels provided needs to be equal to the amount of convolutional layers")

    z_dimension_str = config.get('architecture', 'z_dimension')
    z_dimension = int(z_dimension_str)


    epochs_str = config.get('hyperparameters', 'epochs')
    epochs = int(epochs_str)

    batch_size_str = config.get('hyperparameters', 'batch_size')
    batch_size = int(batch_size_str)

    if variation == 'B-VAE':
        beta_str = config.get('hyperparameters', 'beta')
        beta = int(beta_str)

    configuration = {
        "dataset": dataset,
        "path": path
    }

    architecture = {
        "conv_layers": conv_layers,
        "conv_channels": conv_channels,
        "conv_kernel_sizes": conv_kernel_sizes,
        "conv_strides": conv_strides,
        "conv_paddings": conv_paddings,
        "z_dimension": z_dimension
    }

    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size
    }

    if variation == 'B-VAE':
        hyperparameters["beta"] = beta


    return (configuration, architecture, hyperparameters)


def load_dataset(dataset):
    ds_prompt = "\nEnter the path you want the dataset to be stored.\nIf you already have downloaded it, enter its path\n"
    if (dataset == "MNIST"):
        ds_path = input(ds_prompt)

        train_ds = torchvision.datasets.MNIST(root=ds_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
        test_ds = torchvision.datasets.MNIST(root=ds_path, train=False, download=True, transform=torchvision.transforms.ToTensor())

        return (train_ds, test_ds)
