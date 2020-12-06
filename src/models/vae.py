"""
Implements the Original Variational Autoencoder paper: https://arxiv.org/pdf/1312.6114.pdf

Notation used:

    N: Batch Size
    C: Number of Channels
    H: Height (of picture)
    W: Width (of picture)
    z_dim: The dimension of the latent vector

"""
import sys
sys.path.append("../utils")

import multiprocessing

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from model_utils import *
from utils import *

class VAE(pl.LightningModule):
    """ Class that implements a Variational Autoencoder """

    def __init__(self, architecture, hyperparameters, dataset_info):
        """
        :param dict architecture:      A dictionary containing the hyperparameters that define the
                                         architecture of the model.
        :param dict hyperparameters:   A tuple that corresponds to the shape of the input.
        :param dict dataset_info:      The dimension of the latent vector z (bottleneck).

        The constructor of the Variational Autoencoder.
        """

        # call the constructor of the super class
        super(VAE, self).__init__()

        # initialize class variables regarding the architecture of the model
        self.conv_layers = architecture["conv_layers"]
        self.conv_channels = architecture["conv_channels"]
        self.conv_kernel_sizes = architecture["conv_kernel_sizes"]
        self.conv_strides = architecture["conv_strides"]
        self.conv_paddings = architecture["conv_paddings"]
        self.z_dim = architecture["z_dimension"]

        # unpack the "hyperparameters" dictionary
        self.batch_size = hyperparameters["batch_size"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.scheduler_step_size = hyperparameters["epochs"] // 2

        # unpack the "dataset_info" dictionary
        self.dataset_method = dataset_info["ds_method"]
        self.dataset_shape = dataset_info["ds_shape"]
        self.dataset_path = dataset_info["ds_path"]

        # build the encoder
        self.encoder, self.encoder_shapes = create_encoder(architecture, self.dataset_shape)

        # compute the length of the output of the decoder once it has been flattened
        in_features = self.conv_channels[-1] * np.prod(self.encoder_shapes[-1][:])
        # now define the mean and standard deviation layers
        self.mean_layer = nn.Linear(in_features=in_features, out_features=self.z_dim)
        self.std_layer = nn.Linear(in_features=in_features, out_features=self.z_dim)

        # use a linear layer for the input of the decoder
        in_channels = self.conv_channels[-1]
        self.decoder_input = nn.Linear(in_features=self.z_dim, out_features=in_features)

        # build the decoder
        self.decoder = create_decoder(architecture, self.encoder_shapes)

        # build the output layer
        self.output_layer = create_output_layer(architecture, self.dataset_shape)

    def _encode(self, X):
        """
        :param Tensor X:  Input to encode into mean and standard deviation. (N, C, H, W)

        :return:  The mean and std tensors that the encoder produces for input X. (N, z_dim)
        :rtype:   (Tensor, Tensor)

        This method applies forward propagation to the self.encoder in order to get the mean and
        standard deviation of the latent vector z.
        """
        # run the input through the encoder part of the Network
        encoded_input = self.encoder(X)

        # flatten so that it can be fed to the mean and standard deviation layers
        encoded_input = torch.flatten(encoded_input, start_dim=1)

        # compute the mean and standard deviation
        mean = self.mean_layer(encoded_input)
        std = self.std_layer(encoded_input)

        return mean, std

    def _compute_latent_vector(self, mean, std):
        """
        :param Tensor mean:  The mean of the latent vector z following a Gaussian distribution.
                               (N, z_dim)
        :param Tensor std:   The standard deviation of the latent vector z following a Gaussian
                               distribution. (N, z_dim)

        :return:  The Linear combination of the mean and standard deviation, where the latter
                    factor is multiplied with a random variable epsilon ~ N(0, 1). Basically
                    the latent vector z. (N, z_dim)
        :rtype:   Tensor

        This method computes the latent vector z by applying the reparameterization trick to the
        output of the mean and standard deviation layers, in order to be able to later compute the
        gradient. The stochasticiy here is introduced by the factor epsilon, which is an independent
        node. Thus, we do not have to compute its gradient during backpropagation.
        """

        # compute the stochastic node epsilon
        epsilon = torch.randn_like(std)

        # compute the linear combination of the above attributes and return
        return mean + epsilon * (1.0 / 2) * std

    def _decode(self, z):
        """
        :param Tensor z:  Latent vector computed using the mean and variance layers (with the
                            reparameterization trick). (N, z_dim)

        :return:  The output of the decoder part of the network. (N, C, H, W)
        :rtype:   Tensor

        This method performs forward propagation of the latent vector through the decoder of the
        VAE to get the final output of the network.
        """
        # run the latent vector through the "input decoder" layer
        decoder_input = self.decoder_input(z)

        # convert back the shape that will be fed to the decoder
        height = self.encoder_shapes[-1][0]
        width = self.encoder_shapes[-1][1]
        decoder_input = decoder_input.view(-1, self.conv_channels[-1], height, width)

        # run through the decoder
        decoder_output = self.decoder(decoder_input)

        # run through the output layer and return
        network_output = self.output_layer(decoder_output)
        return network_output

    def forward(self, X):
        """
        :param Tensor X:  The input to run through the VAE. (N, C, H, W)

        :return: The output of the Network, along with the mean, standard deviation layers.
                   (N, C, H, W), (N, z_dim), (N, z_dim)
        :rtype:  (Tensor, Tensor, Tensor)

        This method performs Forward Propagation through all the layers of the VAE and returns
        the reconstructed input.
        """
        # encode the input to get mean and standard deviation
        mean, std = self._encode(X)

        # get the latent vector z by using the reparameterization trick
        z = self._compute_latent_vector(mean, std)

        # compute the output by propagating the latent vector through the decoder and return
        decoded_output = self._decode(z)
        return decoded_output, mean, std

    def configure_optimizers(self):
        """
        :return:  The optimizer that will be used during backpropagation, along with a scheduler.
        :rtype:   (torch.optim.Optimizer, torch.optim.lr_scheduler)
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size,
                                                    gamma=0.1)

        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        """
        :param Tensor batch:   The current batch of the training set.
        :param int batch_idx:  The batch index of the current batch.

        :return:  A dictionary of the losses computed on the current prediction.
        :rtype:   (dict)
        """
        # unpack the current batch
        X, y = batch

        # pass it through the model
        X_hat, mean, std = self(X)
        # calculate the losses
        losses = VAE.criterion(X, X_hat, mean, std)

        return losses

    def train_dataloader(self):
        """
        :return:  A DataLoader object of the training set.
        :rtype:   torch.utils.data.DataLoader
        """
        # download the training set using torchvision if it hasn't already been downloaded
        train_set = self.dataset_method(root=self.dataset_path, train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
        # initialize a pytorch DataLoader to feed training batches into the model
        self.train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True,
                                       num_workers=multiprocessing.cpu_count()//2)
        return self.train_loader

    def test_step(self, batch, batch_idx):
        """
        :param Tensor batch:   The current batch of the test set.
        :param int batch_idx:  The batch index of the current batch.

        :return:  A tuple consisting of a dictionary of the losses computed on the current
                    prediction and the MSE Loss compared to the original picture.
        :rtype:  (dict, Tensor)
        """
        # unpack the current batch
        X, y = batch

        # pass it through the model
        X_hat, mean, std = self(X)
        # calculate the losses
        losses = VAE.criterion(X, X_hat, mean, std)

        # also calculate the MSE loss
        mse_loss_func = torch.nn.MSELoss()
        mse_loss = mse_loss_func(X, X_hat)

        self.log('mse_loss', mse_loss.item())
        self.log('losses', losses)
        return losses, mse_loss

    def test_dataloader(self):
        """
        :return:  A DataLoader object of the test set.
        :rtype:   torch.utils.data.DataLoader
        """
        # download the test set using torchvision if it is not already downloaded
        test_set = self.dataset_method(root=self.dataset_path, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
        # initialize a pytorch DataLoader to feed test batches into the model
        self.test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=True,
                                      num_workers=multiprocessing.cpu_count() // 2)
        return self.test_loader

    def reconstruct(self, n):
        """
        :param int n:  The number of images to plot in each row of whole plot.

        :return:  None

        This method plots n^2 reconstructed (from the test set) images next to each other.
        """

        # get as many batches from the test set to fill the final plot
        tensors = []
        img_count = 0
        while n * n > img_count:
            batch, y = next(iter(self.test_loader))
            img_count += len(batch)
            tensors.append(batch)

        # concatenate them
        X = torch.cat(tensors, dim=0)

        # pass them through the model
        X_hat, mean, std = self(X)
        min_imgs = min(n, len(X))

        # set the correct colourmap that corresponds to the image dimension
        cmap = None
        if (self.dataset_shape[0] == 3):
            cmap = 'viridis'
        elif (self.dataset_shape[0] == 1):
            cmap = 'gray'

        # plot the images and their reconstructions
        plot_multiple(X_hat.detach().numpy(), min_imgs, self.dataset_shape, cmap)

    @staticmethod
    def _data_fidelity_loss(X, X_hat, eps=1e-10):
        """
        :param Tensor X:      The original input data that was passed to the VAE.
                                (N, C, H, W)
        :param Tensor X_hat:  The reconstructed data, the output of the VAE.
                                (N, C, H, W)
        :param Double eps:    A small positive double used to ensure we don't get log of 0.

        :return:  A tensor containing the Data Fidelity term of the loss function,
                    which is given by the formula below.
        :rtype:   Tensor

        E_{z ~ q_{phi}(z | x)}[log(p_{theta}(x|z))] = sum(x * log(x_hat) + (1 - x) * log(1 - x_hat))

            which is basically a Cross Entropy Loss.

        This method computes the Data Fidelity term of the loss function. A small positive double
        epsilon is added inside the logarithm to make sure that we don't get log(0).
        """
        # compute the data fidelity for every training example
        data_fidelity = torch.sum(X * torch.log(eps + X_hat) + (1 - X) * torch.log(eps + 1 - X_hat),
                                  axis=[1, 2, 3])
        return data_fidelity

    @staticmethod
    def _kl_divergence_loss(mean, std):
        """
        :param Tensor mean:  The output of the mean layer, computed with the output of the
                               encoder. (N, z_dim)
        :param Tensor std:   The output of the standard deviation layer, computed with the output
                               of the encoder. (N, z_dim)

        :return:  A tensor consisting of the KL-Divergence term of the loss function, which is
                    given by the formula below.
        :rtype:   Tensor

        D_{KL}[q_{phi}(z | x) || p_{theta}(x)] = (1/2) * sum(std + mean^2 - 1 - log(std))

            In the above equation we substitute std with e^{std} to improve numerical stability.

        This method computes the KL-Divergence term of the loss function. It substitutes the
        value of the standard deviation layer with exp(standard deviation) in order to ensure
        numerical stability.
        """
        # compute the kl divergence for each training example and return it
        kl_divergence = (1 / 2) * torch.sum(torch.exp(std) + torch.square(mean) - 1 - std, axis=1)
        return kl_divergence

    @staticmethod
    def criterion(self, X, X_hat, mean, std):
        """
        :param Tensor X:      The original input data that was passed to the VAE.
                                (N, C, H, W)
        :param Tensor X_hat:  The reconstructed data, the output of the VAE.
                                (N, C, H, W)
        :param Tensor mean:   The output of the mean layer, computed with the output of the
                                encoder. (N, z_dim)
        :param Tensor std:    The output of the standard deviation layer, computed with the output
                                of the encoder. (N, z_dim)

        :return: A dictionary containing the values of the losses computed.
        :rtype:  dict

        This method computes the loss of the VAE using the formula:

            L(x, x_hat) = - E_{z ~ q_{phi}(z | x)}[log(p_{theta}(x|z))]
                          + D_{KL}[q_{phi}(z | x) || p_{theta}(x)]

        Intuitively, the expectation term is the Data Fidelity term, and the second term is a
        regularizer that makes sure the distribution of the encoder and the decoder stay close.
        """
        # get the 2 losses
        data_fidelity_loss = VAE._data_fidelity_loss(X, X_hat)
        kl_divergence_loss = VAE._kl_divergence_loss(mean, std)

        # add them to compute the loss for each training example in the mini batch
        loss = -data_fidelity_loss + kl_divergence_loss

        # place them all inside a dictionary and return it
        losses = {"data_fidelity": torch.mean(data_fidelity_loss),
                  "kl-divergence": torch.mean(kl_divergence_loss),
                  "loss": torch.mean(loss)}
        return losses
