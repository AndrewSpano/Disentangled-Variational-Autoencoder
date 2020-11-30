"""
Implements the betaVAE from the paper: https://arxiv.org/pdf/1606.05579.pdf
"""
import sys
sys.path.append("../../utils")

import torch
import torch.nn as nn

from vae import VAE
from model_utils import *
from utils import *


class betaVAE(VAE):
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

        # invoke the constructor of the VAE class, as the architecture is the same
        super(betaVAE, self).__init__(architecture, input_shape, z_dimension)

        # store the value of beta in the class
        self.beta = beta
