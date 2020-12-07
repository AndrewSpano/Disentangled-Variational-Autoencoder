from argparse import ArgumentParser
from configparser import ConfigParser
import torchvision

from utils import str_to_int_list, str_to_tuple_list

def parse_cmd_args(arg=None):
    """
    :param list arg:    A list with all the command line arguments given by the user

    :return:            An object containing all arguments given as attributes

    Function used to parse the command line input of the program
    """

    # create the argument parser
    description = "Python script that creates a variational autoencoder"
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
    """
    :param str filepath:    The path to the config file given as a cmd line argument

    :return:                A tuple containing three dictionaries:
                            Configuration contains general information,
                            Architecture contains information about the
                            architecture of the model,
                            Hyperparameters contains information regarding the
                            hyperparameters used in the model

    Function used to parse the config file given by the user
    """

    # Initialize the dictionaries
    configuration = {}
    architecture = {}
    hyperparameters = {}

    # Initialize the ConfigParser
    config = ConfigParser()
    config.read(filepath)

    # Get the configuration information
    dataset = config.get('configuration', 'dataset')
    path = config.get('configuration', 'path')

    # Get the architecture information
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

    # Get the hyperparameters information
    epochs_str = config.get('hyperparameters', 'epochs')
    epochs = int(epochs_str)

    batch_size_str = config.get('hyperparameters', 'batch_size')
    batch_size = int(batch_size_str)

    lr_str = config.get('hyperparameters', 'learning_rate')
    learning_rate = float(lr_str)

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
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    if variation == 'B-VAE':
        hyperparameters["beta"] = beta


    return (configuration, architecture, hyperparameters)
