import sys
sys.path.append("../utils")

import torch
import torch.nn as nn

from errors import InvalidArchitectureError


def compute_output_shape(current_shape, kernel_size, stride, padding):
    """
    :param tuple current_shape:  The current shape of the data before a convolution is applied.
    :param tuple kernel_size:    The kernel size of the current convolution operation.
    :param tuple stride:         The stride of the current convolution operation.
    :param tuple padding:        The padding of the current convolution operation.

    :return:  The shape after a convolution operation with the above parameters is applied.
    :rtype:   tuple

            The formula used to compute the final shape is

        component[i] = floor((N[i] - K[i] + 2 * P[i]) / S[i]) + 1

        where, N = current shape of the data
               K = kernel size
               P = padding
               S = stride
    """
    # get the dimension of the data compute each component using the above formula
    dimensions = len(current_shape)
    return tuple((current_shape[i] - kernel_size[i] + 2 * padding[i]) // stride[i] + 1
                 for i in range(dimensions))


def compute_transpose_output_shape(current_shape, kernel_size, stride, padding):
    """
    :param tuple current_shape:  The current shape of the data before a transpose convolution is
                                   applied.
    :param tuple kernel_size:    The kernel size of the current transpose convolution operation.
    :param tuple stride:         The stride of the current transpose convolution operation.
    :param tuple padding:        The padding of the current transpose convolution operation.

    :return:  The shape after a transpose convolution operation with the above parameters is
                applied.
    :rtype:   tuple

            The formula used to compute the final shape is

        component[i] = (N[i] - 1) * S[i] - 2 * P[i] + (K[i] - 1) + 1

        where, N = current shape of the data
               K = kernel size
               P = padding
               S = stride
    """
    # get the dimension of the data compute each component using the above formula
    dimensions = len(current_shape)
    return tuple((current_shape[i] - 1) * stride[i] - 2 * padding[i] + (kernel_size[i] - 1) + 1
                 for i in range(dimensions))


def compute_output_padding(current_shape, target_shape):
    """
    :param tuple current_shape:  The shape of the data after a transpose convolution operation
                                   takes place.
    :param tuple target_shape:   The target shape that we would like our data to have after the
                                   transpose convolution operation takes place.

    :return:  The output padding needed so that the shape of the image after a transpose
                convolution is applied, is the same as the target shape.
    :rtype:   tuple
    """
    # basically subtract each term to get the difference which will be the output padding
    dimensions = len(current_shape)
    return tuple(target_shape[i] - current_shape[i] for i in range(dimensions))


def invalid_shape(current_shape):
    """
    :param tuple current_shape:  The current shape of the data after a convolution is applied.

    :return:  True if the shape is invalid, that is, a negative or 0 components exists. Else, it
                returns False.
    :rtype:   bool
    """
    # check all components
    for component in current_shape:
        if component <= 0:
            return True
    # return False if they are ok
    return False


def create_encoder(architecture, input_shape):
    """
    :param dict architecture:  A dictionary containing the hyperparameters that define the
                                 architecture of the model.
    :param tuple input_shape:  A tuple that corresponds to the shape of the input.

    :return:  A PyTorch Sequential model that represents the encoder part of a VAE, along with the
                final shape that a data point would have after the sequential is applied to it.
    :rtype:   (torch.nn.Sequential, tuple)

    This method builds the encoder part of a VAE and returns it. It is common for all types of VAE.
    """

    # initialize useful variables
    in_channels = input_shape[0]
    current_shape = (input_shape[1], input_shape[2])

    # initialize a list that will store the shape produced in each layer
    shape_per_layer = [current_shape]

    # build the encoder part
    conv_sets = []

    # iterate through the lists that define the architecture of the encoder
    for layer in range(architecture["conv_layers"]):

        # get the variables from the dictionary for more verbose
        out_channels = architecture["conv_channels"][layer]
        kernel_size = architecture["conv_kernel_sizes"][layer]
        stride = architecture["conv_strides"][layer]
        padding = architecture["conv_paddings"][layer]

        # add a set of Convolutional - Leaky ReLU - Batch Normalization sequential layers
        conv_sets.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.LeakyReLU(negative_slope=0.15),
                nn.BatchNorm2d(out_channels))
        )

        # compute the new shape of the image
        current_shape = compute_output_shape(current_shape=current_shape,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding)
        shape_per_layer.append(current_shape)

        # make sure that the shape is valid, and if not, raise an error
        if invalid_shape(current_shape):
            raise InvalidArchitectureError(shape=current_shape, layer=layer+1)

        # the output channels of the current layer becomes the input channels of the next layer
        in_channels = out_channels

    # create a Sequential model and return it (* asterisk is used to unpack the list)
    return nn.Sequential(*conv_sets), shape_per_layer


def create_decoder(architecture, encoder_shapes):
    """
    :param dict architecture:    A dictionary containing the hyperparameters that define the
                                   architecture of the model.
    :param list encoder_shapes:  A list that contains the shape of the data after it is applied to
                                    every set of convolutional layers.

    :return:  A PyTorch Sequential model that represents the decoder part of a VAE.
    :rtype:   (torch.nn.Sequential)

    This method builds the decoder part of a VAE and returns it. It is common for all types of VAE.
    """
    # now start building the decoder part
    conv_sets = []

    # initialize useful variables
    in_channels = architecture["conv_channels"][-1]

    # iterate through the lists that define the architecture of the decoder
    for layer in range(architecture["conv_layers"] - 1, -1, -1):

        # get the variables from the dictionary for more verbose
        out_channels = architecture["conv_channels"][layer]
        kernel_size = architecture["conv_kernel_sizes"][layer]
        stride = architecture["conv_strides"][layer]
        padding = architecture["conv_paddings"][layer]

        # compute the output shape after a transpose convolution in order to get the output padding
        current_shape = encoder_shapes[layer + 1]
        target_shape = encoder_shapes[layer]
        output_shape = compute_transpose_output_shape(current_shape=current_shape,
                                                      kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=padding)
        output_padding = compute_output_padding(output_shape, target_shape)

        # add a set of ConvolutionalTranspose - Leaky ReLU - Batch Normalization sequential layers
        conv_sets.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=output_padding),
                nn.LeakyReLU(negative_slope=0.15),
                nn.BatchNorm2d(out_channels))
        )

        # the output channels of the current layer becomes the input channels of the next layer
        in_channels = out_channels

    # create a Sequential model and return it (* asterisk is used to unpack the list)
    return nn.Sequential(*conv_sets)


def create_output_layer(architecture, input_shape):
    """
    :param dict architecture:  A dictionary containing the hyperparameters that define the
                                 architecture of the model.
    :param tuple input_shape:  A tuple that corresponds to the shape of the input.

    :return:  A PyTorch Sequential model that represents the output layer of a VAE.
    :rtype:   torch.nn.Sequential

    This method creates the output layer of a VAE, that is, the layer where the data from the
    output of the decoder gets fed in order to be finally reconstructed.
    """
    # define the variables of the architecture for more verbose
    in_channels = architecture["conv_channels"][0]
    kernel_size = architecture["conv_kernel_sizes"][0]
    stride = architecture["conv_strides"][0]
    padding = architecture["conv_paddings"][0]

    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=in_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                         nn.SELU(),
                         nn.BatchNorm2d(in_channels),
                         nn.Conv2d(in_channels=in_channels,
                                   out_channels=input_shape[0],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                         nn.Sigmoid())
