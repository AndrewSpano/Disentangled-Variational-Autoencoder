import torch
import torch.nn as nn
from torch.nn import functional as F

import sys
sys.path.append("../../utils")

from utils import *


class betaVAE(nn.Module):
    """ Class that implements a Disentangled Variational Autoencoder """

    def __init__(self, architecture, input_shape, z_dimension, beta):
        """
        :param architecture: (dict)  A dictionary containing the hyperparameters that define the
                                     architecture of the model.
        :param input_shape:  (tuple) A tuple that corresponds to the shape of the input.
        :param z_dimension:  (int)   The dimension of the latent vector z (bottleneck).
        :param beta:         (float) The disentanglment factor to be multiplied with the KL
                                     divergence.

        The constructor of the Disentangled Variational Autoencoder.
        """

        # call the constructor of the super class
        super(betaVAE, self).__init__()

        # initialize class variables regarding the architecture of the model
        self.input_shape = input_shape
        self.conv_layers = architecture["conv_layers"]
        self.conv_kernel_sizes = architecture["conv_kernel_sizes"]
        self.strides = architecture["strides"]
        self.channels = architecture["channels"]
        self.z_dim = z_dimension

        # class variables regarding the loss of the model
        self.beta = beta

        # the number of channels that the input image has
        in_channels = self.input_shape[1]

        # build the encoder part
        sets_of_conv_selu_bn = []

        # iterate through the lists that define the architecture of the encoder
        for layer in range(self.conv_layers):

            # add a set of Convolutional - SeLU - Batch Normalization sequential layers
            conv = nn.Conv2d(in_channels=in_channels, out_channels=self.channels[layer],
                             kernel_size=self.conv_kernel_sizes[layer], stride=self.strides[layer],
                             padding=1)
            selu = nn.SELU()
            batch_norm = nn.BatchNorm2d(self.channels[layer])

            # define a sequential model with the above architecture append it to the list
            sets_of_conv_selu_bn.append(nn.Sequential(conv, selu, batch_norm))

            # the output channels of the current layer becomes the input channels of the next layer
            in_channels = self.channels[layer]

        # finish the encoder
        self.encoder = nn.Sequential(*sets_of_conv_selu_bn)

        # now define the mean and standard deviation layers
        self.mean_layer = nn.Linear(in_features=self.channels[-1], out_features=self.z_dim)
        self.std_layer = nn.Linear(in_features=self.channels[-1], out_features=self.z_dim)

        # use a linear layer for the input of the decoder
        in_channels = self.channels[-1]
        self.decoder_input = nn.Linear(in_features=self.z_dim, out_features=in_channels)

        # now start building the decoder part
        sets_of_convtr_selu_bn = []

        # iterate through the lists that define the architecture of the decoder
        for layer in range(self.conv_layers - 1, 0, -1):

            # add a set of DeConvolutional - SeLU - Batch Normalization sequential layers
            convtr = nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.channels[layer],
                                        kernel_size=self.conv_kernel_sizes[layer],
                                        stride=self.strides[layer], padding=1)
            selu = nn.SELU()
            batch_norm = nn.BatchNorm2d(self.channels[layer])

            # define a sequential model with this architecture append it to the list
            sets_of_convtr_selu_bn.append(nn.Sequential(convtr, selu, batch_norm))

            # the output channels of the current layer becomes the input channels of the next layer
            in_channels = self.channels[layer]

        # finish the decoder
        self.decoder = nn.Sequential(*sets_of_convtr_selu_bn)

        # define the output layer: Conv2d -> SELU -> Batch Norm -> Conv2d -> Sigmoid
        self.output_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                             out_channels=in_channels,
                                                             kernel_size=self.conv_kernel_sizes[0],
                                                             stride=self.strides[0], padding=1),
                                          nn.SELU(),
                                          nn.BatchNorm2d(in_channels),
                                          nn.Conv2d(in_channels=in_channels,
                                                    out_channels=self.input_shape[1],
                                                    kernel_size=self.conv_kernel_sizes[0],
                                                    stride=self.strides[0], padding=1),
                                          nn.Sigmoid())

    def encode(self, X):
        """
        :param X: (Tensor) Input to encode into mean and standard deviation.

        :return: (tuple) The mean and std tensors that the encoder produces for input X.

        This method applies forward propagation to the self.encoder in order to get the mean and
        standard deviation of the latent vector z.
        """
        # run the input through the encoder part of the Nerwork
        encoded_input = self.encoder(X)
        # flatten so that it can be fed to the mean and standard deviation layers
        encoded_input = torch.flatten(encoded_input, start_dim=1)

        # compute the mean and standard deviation
        mean = self.mean_layer(encoded_input)
        std = self.std_layer(encoded_input)

        return mean, std

    def reparameterize(self, mean, std):
        """
        :param mean: (Tensor) The mean of the latent vector z following a Gaussian distribution.
        :param std:  (Tensor) The standard deviation of the latent vector z following a Gaussian
                              distribution.

        :return: (Tensor) Linear combination of the mean and standard deviation, where the latter
                          factor is multiplied with a random variable epsilon ~ N(0, 1).

        This method applies the reparameterization trick to the output of the mean and standard
        deviation layers, in order to be able to compute the gradient. The stochasticiy here is
        introduced by the factor epsilon, which is an independent node. Thus, we do not have to
        compute its gradient during backpropagation.
        """

        # compute the stochastic node epsilon
        epsilon = torch.randn_like(std)
        # raise the standard deviation to an exponent, to improve numberical stability
        std = torch.exp(1/2 * std)

        # compute the linear combination of the above attributes and return
        return mean + epsilon * std

    def decode(self, z):
        """
        :param z: (Tensor) Latent vector computed using the mean and variance layers (with the
                           reparameterization trick).

        :return: (Tensor) The output of the decoder part of the network.

        This method performs forward propagation of the latent vector through the decoder of the
        betaVAE to get the final output of the network.
        """
        # run the latent vector through the "input decoder" layer
        decoder_input = self.decoder_input(z)

        # convert back the shape that will be fed to the decoder
        decoder_input = decoder_input.view(-1, self.conv_kernel_sizes[-1], 1, 1)

        # run through the decoder
        decoder_output = self.decoder(decoder_input)

        # run through the output layer and return
        network_output = self.output_layer(decoder_output)
        return network_output

    def forward(self, X):
        """
        """
        pass
