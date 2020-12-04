# Disentangled Variational Autoencoder

PyTorch Implementation of the papers
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
<br> </br>


## Structure of Repo

- The directory [src/models](src/models) contains the models we have created so far. More are coming along the way.
- The python script [src/main.py](src/main.py) is used for training and testing.
- In the [mathematical_analysis](mathematical_analysis) directory there is a [pdf](mathematical_analysis/vae_maths.pdf) where the basic mathematical concepts of the Variational Autoencoder are explained thoroughly.
- In the [config](config) directory there are some configuration files that can be used to create the models.
<br> </br>


## Models
Currently two models are supported, a simple Variational Autoencoder and a Disentangled version (beta-VAE). The model implementations can be found in the [src/models](src/models) directory. These models were developed using [PyTorch Lightning](https://www.pytorchlightning.ai/).

- [Variational Autoencdoer](src/models/vae.py)
    - <u>What it is</u>
     ### TODO Andreas
    - <u>How it is implemented</u>
      The model is implemented through a class which inherits from the "basic" lighnting module. At initialization, the encoder, decoder and latent vector layers         are created. Once the fit method of the lightning trainer is called, the model starts training by using the training_step, criterion and forward methods. The       same applies on testing with the respective methods. 
- [Disentangled Variational Autoencoder](src/models/beta_vae.py)
    - <u>What it is</u>
      ### TODO Andreas
    - <u>How it is implemented</u>
      As all other models, this model is represented by a class, this class inherits from original VAE class. The only addition is of course "beta". Due to that           change, only the default constructor and the criterion methods have changed slightly
      
      
## Execution
To execute the VAE, you have to navigate to the [src](src) directory and execute the following command
    ```bash
    $ python3 main.py -c <config_file_path> -v <variation>
    ```
### Configuration File
The configuration file should have the following format:

    ```
    [configuration]
    dataset = <MNIST or CIFAR10>
    path = <path_to_dataset>
    [architecture]
    conv_layers = <amount of convolutional layers>
    conv_channels = <the convolutional channels as a list (e.g. [16,32,64]) of size "conv_layers">
    conv_kernel_sizes = <the convolutional kernel sizes as a list of tuples (e.g. [(7, 7), (5, 5), (3, 3)]) of size "conv_layers">
    conv_strides = <the convolutional strides as a list of tuples (e.g. [(1, 1), (1, 1), (1, 1)]) of size "conv_layers">
    conv_paddings = <the convolutional paddings as a list of tuples (e.g. [(1, 1), (1, 1), (1, 1)]) of size "conv_layers">
    z_dimension = <the dimension of the latent vector>
    [hyperparameters]
    epochs = <self-explanatory>
    batch_size = <self-explanatory>
    learning_rate = <starting learning rate, stepLr is used>
    beta = <beta parameter for B-VAE>
    ```
    
### Variation
Variation indicates what kind of Variational Autoencoder you want to train. Right now, the only options are
    ```bash
    $ python3 main.py -c <config_file_path> -v VAE
    ```
    and
    ```bash
    $ python3 main.py -c <config_file_path> -v B-VAE
    ```
Keep in mind that for the second option, the beta parameter is required in the configuration file (in the section "hyperparameters")
    
