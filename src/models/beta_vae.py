"""
Implements the betaVAE from the paper: https://arxiv.org/pdf/1606.05579.pdf

Notation used:

    N: Batch Size
    C: Number of Channels
    H: Height (of picture)
    W: Width (of picture)
    z_dim: The dimension of the latent vector

"""
import sys
sys.path.append("../utils")

import torch
import torch.nn as nn

from vae import VAE
from model_utils import *
from utils import *


class betaVAE(VAE):
    """
    Class that implements a Disentangled Variational Autoencoder (Beta VAE).
    This class inherits from the VAE class.
    """

    def __init__(self, architecture, hyperparameters, dataset_info):
        """
        :param dict architecture:     A dictionary containing the hyperparameters that define the
                                        architecture of the model.
        :param dict hyperparameters:  A tuple that corresponds to the shape of the input.
        :param dict dataset_info:     The dimension of the latent vector z (bottleneck).

        The constructor of the Disentangled Variational Autoencoder.
        """

        # invoke the constructor of the VAE class, as the architecture is the same
        super(betaVAE, self).__init__(architecture, hyperparameters, dataset_info)

        # store the value of beta in the class as it exists only in this VAE variation
        self.beta = hyperparameters["beta"]

    @staticmethod
    def criterion(X, X_hat, mean, std):
        """
        :param Tensor X:      The original input data that was passed to the B-VAE.
                                (N, input_shape[1], H, W)
        :param Tensor X_hat:  The reconstructed data, the output of the B-VAE.
                                (N, input_shape[1], H, W)
        :param Tensor mean:   The output of the mean layer, computed with the output of the
                                encoder. (N, z_dim)
        :param Tensor std:    The output of the standard deviation layer, computed with the output
                                of the encoder. (N, z_dim)

        :return:  A dictionary containing the values of the losses computed.
        :rtype:   dict

        This method computes the loss of the B-VAE using the formula:

            L(x, x_hat) = - E_{z ~ q_{phi}(z | x)}[log(p_{theta}(x|z))]
                          + beta * D_{KL}[q_{phi}(z | x) || p_{theta}(x)]

        Intuitively, the expectation term is the Data Fidelity term, and the second term is a
        regularizer that makes sure the distribution of the encoder and the decoder stay close.
        """
        # get the 2 losses
        data_fidelity_loss = VAE._data_fidelity_loss(X, X_hat)
        kl_divergence_loss = VAE._kl_divergence_loss(mean, std)

        # add them to compute the loss for each training example in the mini batch
        loss = -data_fidelity_loss + self.beta * kl_divergence_loss

        # place them all inside a dictionary and return it
        losses = {"data_fidelity": torch.mean(data_fidelity_loss),
                  "kl-divergence": torch.mean(kl_divergence_loss),
                  "beta_kl-divergence": self.beta * torch.mean(kl_divergence_loss),
                  "loss": torch.mean(loss)}
        return losses
