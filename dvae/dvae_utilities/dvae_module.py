import torch
import torch.nn as nn
from torch.nn import functional as F

import sys
sys.path.append("../../utils")

from utils import *


class DVAE(nn.Module):
    """ Class that implements a Disentangled Variational Autoencoder """

    def __init__(self, architecture_hyperparam, latent_vector_dimension=2, beta=1):
        """
        :param architecture_hyperparam: (dict)  A dictionary containing the hyperparameters that
                                                define the architecture of the model.
        :param latent_vector_dimension: (int)   The dimension of the latent vector z (bottleneck).
        :param beta:                    (float) The disentanglment factor to be multiplied with the
                                                KL divergence.

        The constructor of the Disentangled Variational Autoencoder.
        """

        # call the constructor of the super class
        super(DVAE, self).__init__()

        # initialize class variables regarding the architecture of the model
        self.conv_layers = architecture_hyperparam["conv_layers"]
        self.conv_kernel_sizes = architecture_hyperparam["conv_kernel_sizes"]
        self.pool_kernel_sizes = architecture_hyperparam["pool_kernel_sizes"]
        self.channels = architecture_hyperparam["channels"]
        self.z_dim = latent_vector_dimension

        # class variables regarding the loss of the model
        self.beta = beta

        # the first input image has only 1 channel (28 x 28 x 1, for MNIST)
        in_channels = 1
        # build the encoder part
        sets_of_layers = []

        # iterate through the lists that define the architecture of the encoder
        for conv_layer in range(self.conv_layers):

            # determine the padding dimension so that we end up with "same" padding
            same_padding = get_same_padding(self.conv_kernel_sizes[conv_layer])

            # add a set of Convolutional - SeLU - Batch Normalization sequential layers
            conv = nn.Conv2d(in_channels=in_channels, out_channels=self.channels[conv_layer],
                             kernel_size=self.conv_kernel_sizes[conv_layer], stride=(1, 1),
                             padding=same_padding)
            selu = nn.SELU()
            batch_norm = nn.BatchNorm2d(self.channels[conv_layer])

            # group the layers in a list
            layers = [conv, selu, batch_norm]

            # check if we should add a MaxPooling layer
            if conv_layer < len(self.pool_kernel_sizes):
                max_pool = nn.MaxPool2d(kernel_size=self.pool_kernel_sizes[conv_layer])
                layers.append(max_pool)

            # define a sequential model with this architecture
            sequential = nn.Sequential(*layers)

            # append it to the set of all the sequentials
            sets_of_layers.append(sequential)

            # the number of input channels for the next layer becomes the number of the current
            # output channels
            in_channels = self.channels[conv_layer]

        # finish the encoder
        self.encoder = nn.Sequential(*sets_of_layers)

        # now define the mean and variance layers
        self.mean_layer = nn.Linear(in_features=self.channels[-1], out_features=self.z_dim)
        self.variance_layer = nn.Linear(in_features=self.channels[-1], out_features=self.z_dim)


        # use a linear layer for the input of the decoder
        in_channels = self.channels[-1]
        self.decoder_input = nn.Linear(in_features=self.z_dim, out_features=in_channels)
        # now start building the decoder part
        sets_of_layers = []

        # iterate through the lists that define the architecture of the decoder
        for conv_layer in range(self.conv_layers - 1, -1, -1):

            # determine the padding dimension so that we end up with "same" padding
            same_padding = get_same_padding(self.conv_kernel_sizes[conv_layer])

            # add a set of Convolutional - SeLU - Batch Normalization sequential layers
            conv = nn.Conv2d(in_channels=in_channels, out_channels=self.channels[conv_layer],
                             kernel_size=self.conv_kernel_sizes[conv_layer], stride=(1, 1),
                             padding=same_padding)
            selu = nn.SELU()
            batch_norm = nn.BatchNorm2d(self.channels[conv_layer])

            # group the layers in a list
            layers = [conv, selu, batch_norm]

            # check if we should add an Upsample layer
            if conv_layer < len(self.pool_kernel_sizes):
                up_sample = nn.Upsample(size=self.pool_kernel_sizes[conv_layer], mode="bilinear")
                layers.append(up_sample)

            # define a sequential model with this architecture
            sequential = nn.Sequential(*layers)

            # append it to the set of all the sequentials
            sets_of_layers.append(sequential)

            # the number of input channels for the next layer becomes the number of the current
            # output channels
            in_channels = self.channels[conv_layer]

        # finish the decoder
        self.decoder = nn.Sequential(*sets_of_layers)

        # define the output layer: Conv2d -> SELU -> Batch Norm -> Conv2d -> Sigmoid
        same_padding = get_same_padding(self.conv_kernel_sizes[0])
        self.output_layer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=in_channels,
                                                    kernel_size=self.conv_kernel_sizes[0],
                                                    stride=(1, 1), padding=same_padding),
                                          nn.SELU(),
                                          nn.BatchNorm2d(in_channels),
                                          nn.Conv2d(in_channels=in_channels,
                                                    out_channels=1,
                                                    kernel_size=self.conv_kernel_sizes[0],
                                                    stride=(1, 1), padding=same_padding),
                                          nn.Sigmoid())

    def encode(self, X):
        """
        :param X: (Tensor) Input to encode into mean and variance.

        :return: (tuple) The mean and variance tensors that the encoder produces for input X.

        This method applies forward propagation to the self.encoder in order to get the mean and
        variance of the latent vector z.
        """
        # run the input through the sets of Convolutional - SeLU - Batch Normalization layers
        encoded_input = self.encoder(X)
        # flatten
        encoded_input = torch.flatten(encoded_input, start_dim=1)

        # compute the mean and variance
        mean = self.mean_layer(encoded_input)
        variance = self.variance_layer(encoded_input)

        # return them
        return mean, variance

    def decode(self, z):
        """
        :param z: (Tensor) Latent vector computed using the mean and variance layers (with the
                           reparameterization trick).

        :return: (Tensor) The output of the DVAE, the reconstructed image X.

        This method performs forward propagation of the latent vector through the decoder of the
        DVAE to get the final output of the network.
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
