import sys
sys.path.append("../utils")

import torch
import torch.nn as nn

from errors import InvalidArchitectureError


def compute_output_shape(current_shape, kernel_size, stride, padding):
    """
    :param current_shape: (tuple) The current shape of the data before a convolution is applied.
    :param kernel_size:   (tuple) The kernel size of the current convolution operation.
    :param stride:        (tuple) The stride of the current convolution operation.
    :param padding:       (tuple) The padding of the current convolution operation.

    :return: (tuple) The shape after a convolution operation with the above parameters is applied.
                     The formula used to compute the final shape is

        component[i] = floor((W[i] - K[i] + 2 * P[i]) / S[i]) + 1

        where, W = input shape of the data
               K = kernel size
               P = padding
               S = stride
    """
    # get the dimension of the data
    dimensions = len(current_shape)
    # compute each component using the above formula and return
    return tuple((current_shape[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1
                 for i in range(dimensions))


def invalid_shape(current_shape):
    """
    :param current_shape: (tuple) The current shape of the data after a convolution is applied.

    :return: (bool) True is the shape is invalid, that is, a negative or 0 components exists. Else,
                    it returns False.
    """
    # check all components
    for component in current_shape:
        if component <= 0:
            return True
    # return False if they are ok
    return False


def create_encoder(architecture, input_shape):
    """
    :param architecture: (dict)  A dictionary containing the hyperparameters that define the
                                 architecture of the model.
    :param input_shape:  (tuple) A tuple that corresponds to the shape of the input.

    :return: (Sequential) A PyTorch Sequential model that represents the encoder part of a VAE.

    This method builds the encoder part of a VAE and returns it. It is common for all types of VAE.
    """

    # the number of channels that the input image has
    in_channels = input_shape[0]

    # keep track of the current Height and Width of the image
    current_shape = (input_shape[1], input_shape[2])

    # build the encoder part
    sets_of_conv_selu_bn = []

    # iterate through the lists that define the architecture of the encoder
    for layer in range(architecture["conv_layers"]):

        # define the number of output channels (filters) for this layer
        out_channels = architecture["conv_channels"][layer]

        # add a set of Convolutional - SeLU - Batch Normalization sequential layers
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=architecture["conv_kernel_sizes"][layer],
                         stride=architecture["conv_strides"][layer],
                         padding=architecture["conv_paddings"][layer])
        selu = nn.SELU()
        batch_norm = nn.BatchNorm2d(out_channels)

        # define a sequential model with the above architecture append it to the list
        sets_of_conv_selu_bn.append(nn.Sequential(conv, selu, batch_norm))

        # compute the new shape of the image
        current_shape = compute_output_shape(current_shape,
                                             architecture["conv_kernel_sizes"][layer],
                                             stride=architecture["conv_strides"][layer],
                                             padding=architecture["conv_paddings"][layer])

        # make sure that the shape is valid, and if not, raise an error
        if invalid_shape(current_shape):
            # raise InvalidArchitectureError(shape=current_shape, layer=layer + 1)
            raise InvalidArchitectureError(shape=current_shape, layer=layer+1)


        # the output channels of the current layer becomes the input channels of the next layer
        in_channels = out_channels

    # create a Sequential model and return it (* asterisk is used to unpack the list)
    return nn.Sequential(*sets_of_conv_selu_bn), current_shape


def create_decoder(architecture):
    """
    :param architecture: (dict)  A dictionary containing the hyperparameters that define the
                                 architecture of the model.

    :return: (Sequential) A PyTorch Sequential model that represents the decoder part of a VAE.

    This method builds the decoder part of a VAE and returns it. It is common for all types of VAE.
    """
    # now start building the decoder part
    sets_of_convtr_selu_bn = []

    # define the current number of channels (after the reformation of the latent vector z)
    in_channels = architecture["conv_channels"][-1]

    # iterate through the lists that define the architecture of the decoder
    for layer in range(architecture["conv_layers"] - 1, -1, -1):

        # define the number of output channels (filters) for this layer
        out_channels = architecture["conv_channels"][layer]

        # add a set of ConvolutionalTranspose - SeLU - Batch Normalization sequential layers
        convtr = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=architecture["conv_kernel_sizes"][layer],
                                    stride=architecture["conv_strides"][layer],
                                    padding=architecture["conv_paddings"][layer])
        selu = nn.SELU()
        batch_norm = nn.BatchNorm2d(out_channels)

        # define a sequential model with this architecture append it to the list
        sets_of_convtr_selu_bn.append(nn.Sequential(convtr, selu, batch_norm))

        # the output channels of the current layer becomes the input channels of the next layer
        in_channels = out_channels

    # create a Sequential model and return it (* asterisk is used to unpack the list)
    return nn.Sequential(*sets_of_convtr_selu_bn)


def create_output_layer(architecture, input_shape):
    """
    :param architecture: (dict)  A dictionary containing the hyperparameters that define the
                                 architecture of the model.
    :param input_shape:  (tuple) A tuple that corresponds to the shape of the input.

    :return: (Sequential) A PyTorch Sequential model that represents the output layer of a VAE.

    This method creates the output layer of a VAE, that is, the layer where the data from the
    output of the decoder gets fed in order to be finally reconstructed.
    """

    # define the number of input channels of the last layer
    in_channels = architecture["conv_channels"][0]

    # define the output layer: ConvTranspose2d -> SELU -> Batch Norm -> Conv2d -> Sigmoid
    convtr = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=architecture["conv_kernel_sizes"][0],
                                stride=architecture["conv_strides"][0],
                                padding=architecture["conv_paddings"][0])
    selu = nn.SELU()
    batch_norm = nn.BatchNorm2d(in_channels)
    conv = nn.Conv2d(in_channels=in_channels, out_channels=input_shape[0],
                     kernel_size=architecture["conv_kernel_sizes"][0],
                     stride=architecture["conv_strides"][0],
                     padding=architecture["conv_paddings"][0])
    sigmoid = nn.Sigmoid()

    # create a Sequential model and return it
    return nn.Sequential(convtr, selu, batch_norm, conv, sigmoid)
