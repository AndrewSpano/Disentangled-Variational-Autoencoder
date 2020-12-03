"""
Implements the Original Variational Autoencoder paper: https://arxiv.org/pdf/1312.6114.pdf
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

DEFAULT_LEARNING_RATE = 1e-3

class VAE(pl.LightningModule):
    """ Class that implements a Variational Autoencoder """

    def __init__(self, architecture, hyperparameters, dataset_info):
        """
        :param architecture: (dict)  A dictionary containing the hyperparameters that define the
                                     architecture of the model.
        :param input_shape:  (tuple) A tuple that corresponds to the shape of the input.
        :param z_dimension:  (int)   The dimension of the latent vector z (bottleneck).

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

        self.batch_size = hyperparameters["batch_size"]

        self.dataset_method = dataset_info["ds_method"]
        self.dataset_shape = dataset_info["ds_shape"]
        self.dataset_path = dataset_info["ds_path"]

        # build the encoder
        self.encoder, self.encoder_output_shape = create_encoder(architecture, self.dataset_shape)

        # compute the length of the output of the decoder once it has been flattened
        in_features = self.conv_channels[-1] * np.prod(self.encoder_output_shape[:])
        # now define the mean and standard deviation layers
        self.mean_layer = nn.Linear(in_features=in_features, out_features=self.z_dim)
        self.std_layer = nn.Linear(in_features=in_features, out_features=self.z_dim)

        # use a linear layer for the input of the decoder
        in_channels = self.conv_channels[-1]
        self.decoder_input = nn.Linear(in_features=self.z_dim, out_features=in_features)

        # build the decoder
        self.decoder = create_decoder(architecture)

        # build the output layer
        self.output_layer = create_output_layer(architecture, self.dataset_shape)

    def _encode(self, X):
        """
        :param X: (Tensor) Input to encode into mean and standard deviation.
                           (N, input_shape[1], H, W)

        :return: (tuple) The mean and std tensors that the encoder produces for input X. (N, z_dim)

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

    def _compute_latent_vector(self, mean, std):
        """
        :param mean: (Tensor) The mean of the latent vector z following a Gaussian distribution.
                              (N, z_dim)
        :param std:  (Tensor) The standard deviation of the latent vector z following a Gaussian
                              distribution. (N, z_dim)

        :return: (Tensor) Linear combination of the mean and standard deviation, where the latter
                          factor is multiplied with a random variable epsilon ~ N(0, 1). Basically
                          the latent vector z. (N, z_dim)

        This method computes the latent vector z by applying the reparameterization trick to the
        output of the mean and standard deviation layers, in order to be able to later compute the
        gradient. The stochasticiy here is introduced by the factor epsilon, which is an independent
        node. Thus, we do not have to compute its gradient during backpropagation.
        """

        # compute the stochastic node epsilon
        epsilon = torch.randn_like(std)
        # raise the standard deviation to an exponent, to improve numberical stability
        std = torch.exp(1/2 * std)

        # compute the linear combination of the above attributes and return
        return mean + epsilon * std

    def _decode(self, z):
        """
        :param z: (Tensor) Latent vector computed using the mean and variance layers (with the
                           reparameterization trick). (N, z_dim)

        :return: (Tensor) The output of the decoder part of the network. (N, input_shape[1], H, W)

        This method performs forward propagation of the latent vector through the decoder of the
        VAE to get the final output of the network.
        """
        # run the latent vector through the "input decoder" layer
        decoder_input = self.decoder_input(z)

        # convert back the shape that will be fed to the decoder
        height = self.encoder_output_shape[0]
        width = self.encoder_output_shape[1]
        decoder_input = decoder_input.view(-1, self.conv_channels[-1], height, width)

        # run through the decoder
        decoder_output = self.decoder(decoder_input)

        # run through the output layer and return
        network_output = self.output_layer(decoder_output)
        return network_output

    def forward(self, X):
        """
        :param X: (Tensor) The input to run through the VAE. (N, input_shape[1], H, W)

        :return: (tuple) The output of the Network, and the mean-standard deviation layers.
                         (N, input_shape[1], H, W), (N, z_dim), (N, z_dim)

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
        return torch.optim.Adam(self.parameters(), lr=DEFAULT_LEARNING_RATE)


    def training_step(self, batch, batch_idx):
        X, y = batch

        output, mean, std = self(X)
        losses = self.criterion(X, output, mean, std)

        return losses

    def train_dataloader(self):
        train_set_path = self.dataset_path
        train_set = self.dataset_method(root=train_set_path, train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())

        self.train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//2)
        return self.train_loader

    def test_step(self, batch, batch_idx):
        X, y = batch

        X_hat, mean, std = self(X)

        losses = self.criterion(X, X_hat, mean, std)

        mse_loss_func = torch.nn.MSELoss()
        mse_loss = mse_loss_func(X, X_hat)

        self.log('mse_loss', mse_loss.item())
        self.log('losses', losses)
        return losses, mse_loss

    def test_dataloader(self):
        test_set_path = self.dataset_path
        test_set = self.dataset_method(root=test_set_path, train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())

        self.test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()//2)
        return self.test_loader

    def sample(self, number_of_images):
        X, y = next(iter(self.test_loader))

        output, mean, std = self(X)

        max_imgs = min(number_of_images, len(X))
        for i in range(max_imgs):
            if (self.dataset_shape[0] == 3):
                X_np = X[i].detach().numpy().ravel()
                X_hat_np = output[i].detach().numpy().ravel()

                X_np = np.reshape(X_np, (self.dataset_shape[1], self.dataset_shape[2], 3), order='F')
                X_hat_np = np.reshape(X_hat_np, (self.dataset_shape[1], self.dataset_shape[2], 3), order='F')

                X_np = np.rot90(X_np, 3)
                X_hat_np = np.rot90(X_hat_np, 3)
                # X_np = np.rot90(X_np)
                # X_hat_np = np.rot90(X_hat_np)
                # X_np = np.rot90(X_np)
                # X_hat_np = np.rot90(X_hat_np)

                plot_against(X_np, X_hat_np, y[i].item(), 'viridis')
            elif (self.dataset_shape[0] == 1):
                plot_against(X[i][0].detach().numpy(), output[i][0].detach().numpy(), y[i].item(), 'gray')

    @staticmethod
    def _data_fidelity_loss(X, X_hat, eps=1e-10):
        """
        :param X:     (Tensor) The original input data that was passed to the VAE.
                               (N, input_shape[1], H, W)
        :param X_hat: (Tensor) The reconstructed data, the output of the VAE.
                               (N, input_shape[1], H, W)
        :param eps:   (Double) A small positive double used to ensure we don't get log of 0.

        :return: (Tensor) The Data Fidelity term of the loss function, which is given by the formula

            E_{z ~ Q_{phi}(z | x)}[log(P_{theta}(x|z))] =

                sum(x * log(x_hat) + (1 - x) * log(1 - x_hat))

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
        :param mean:  (Tensor) The output of the mean layer, computed with the output of the
                               encoder. (N, z_dim)
        :param std:   (Tensor) The output of the standard deviation layer, computed with the output
                               of the encoder. (N, z_dim)

        :return: (Tensor) The KL-Divergence term of the loss function, which is given by the formula

            D_{KL}[Q_{phi}(z | x) || P_{theta}(x)] =

                (1/2) * sum(std + mean^2 - 1 - log(std))

            In the above equation we substitute std with e^{std} to improve numerical stability.

        This method computes the KL-Divergence term of the loss function. It substitutes the
        value of the standard deviation layer with exp(standard deviation) in order to ensure
        numerical stability.
        """
        # compute the kl divergence for each training example and return it
        kl_divergence = (1 / 2) * torch.sum(torch.exp(std) + torch.square(mean) - 1 - std, axis=1)
        return kl_divergence

    def criterion(self, X, X_hat, mean, std):
        """
        :param X:     (Tensor) The original input data that was passed to the VAE.
                               (N, input_shape[1], H, W)
        :param X_hat: (Tensor) The reconstructed data, the output of the VAE.
                               (N, input_shape[1], H, W)
        :param mean:  (Tensor) The output of the mean layer, computed with the output of the
                               encoder. (N, z_dim)
        :param std:   (Tensor) The output of the standard deviation layer, computed with the output
                               of the encoder. (N, z_dim)

        :return: (Dict) A dictionary containing the values of the losses computed.

        This method computes the loss of the VAE using the formula:

            L(x, x_hat) = - E_{z ~ Q_{phi}(z | x)}[log(P_{theta}(x|z))]
                          + D_{KL}[Q_{phi}(z | x) || P_{theta}(x)]

        Intuitively, the expectation term is the Data Fidelity term, and the second term is a
        regularizer that makes sure the distribution of the encoder and the decoder stay close.
        """
        # get the 2 losses
        data_fidelity_loss = VAE._data_fidelity_loss(X, X_hat)
        kl_divergence_loss = VAE._kl_divergence_loss(mean, std)

        # add them, and then compute the mean over all training examples
        loss = -data_fidelity_loss + kl_divergence_loss
        loss = torch.mean(loss)

        # place them all inside a dictionary and return it
        losses = {"data_fidelity": torch.mean(data_fidelity_loss),
                  "kl-divergence": torch.mean(kl_divergence_loss),
                  "loss": loss}
        return losses
