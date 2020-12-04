# Disentangled Variational Autoencoder

PyTorch Implementation of the papers
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
<br> </br>


## Team members:
- Andreas Spanopoulos (andrewspanopoulos@gmail.com)
- Demetrios Konstantinidis (demetris.konstantinidis@gmail.com)
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

    The Variational Autoencoder is a Generative Model. Its goal is to learn the distribution of a Dataset, and then generate new (unseen) data points from the same distribution. In the picture belowe we can see an overview of its architecture.

    ![image](./mathematical_analysis/images/vae-gaussian.png)

    Note than in our implementation, before the main and std layers,
    convolutions have been applied to reduce the dimensionality of
    the data.

- [beta Variational Autoencoder](src/models/beta_vae.py)
    
    Another form of a Variational Autoencoder is the beta-VAE. The difference between the Vanilla VAE and the beta-VAE is in the loss function of the latter: The KL-Divergence term is multiplied with a hyperprameter beta. This introduces a disentanglement to the idea of the VAE, as in many cases it allows a smoother and more "continious" transition of the output data, for small changes in the latent vector z. More information on this topic can be found in the sources section below.

Note that for a more in-depth explanation of how and why the VAE framework actually makes sense, some custom latex notes have been made in a pdf
[here](mathematical_analysis/vae_maths.pdf).
<br> </br>


## Results of both models here
## ToDo: Dimitris


## Mathematics of VAE
As stated above, I have written in Latex some [notes](mathematical_analysis/vae_maths.pdf) about the mathematics behind the Variational Autoencoder. Initially, I started making these notes for myself. Later I realized that they could assist others in understanding the basic concepts of the VAE. Hence, I made them public in the repo. Special thanks to [Ahlad Kumar](https://www.youtube.com/user/kumarahlad) for explaining complex mathematical topics that were not taught in our Bachelor. Also a big thanks to all the resources I have listed in the end of the repo.
<br> </br>

      
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
conv_channels = <the convolutional channels as a list, e.g. [16,32,64]>
conv_kernel_sizes = <the convolutional kernel sizes as a list of tuples, e.g. [(7, 7), (5, 5), (3, 3)]>
conv_strides = <the convolutional strides as a list of tuples, e.g. [(1, 1), (1, 1), (1, 1)]>
conv_paddings = <the convolutional paddings as a list of tuples, e.g. [(1, 1), (1, 1), (1, 1)]>
z_dimension = <the dimension of the latent vector>
[hyperparameters]
epochs = <e.g. 10>
batch_size = <e.g. 32>
learning_rate = <starting learning rate, e.g. 0.000001>
beta = <beta parameter for B-VAE, e.g. 4>
```

Examples of configuration files can be found in the [config](config) directory.
    
### Variation
Variation indicates what kind of Variational Autoencoder you want to train. Right now, the only options are:
1. Vanilla VAE: 
    ```bash
    $ python3 main.py -c <config_file_path> -v VAE
    ```
2. beta VAE:
    ```bash
    $ python3 main.py -c <config_file_path> -v B-VAE
    ```
Keep in mind that for the second option, the beta parameter is required in the configuration file (in the section "hyperparameters").
<br> </br>


## Colab Notebook
Under the [notebook](notebook) directory exists a colab notebook. This is useful for experimentation/testing as it makes the usage of a GPU much easier.
Keep in mind that the core code remains the same, however the initialization of values is done manually.
<br> </br>


## Sources
- Papers:
    - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
    - [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)
    - [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
    - [Early Visual Concept Learningwith Unsupervised Deep Learning](https://arxiv.org/pdf/1606.05579.pdf)
- Online Articles:
    - [TowardsDataScience](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
    - [Jaan Altosaar](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
- Online Lectures:
    - [Ahlad Kumar](https://www.youtube.com/watch?v=w8F7_rQZxXk&list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe)
    - [Ali Ghodsi](https://www.youtube.com/watch?v=uaaqyVS9-rM&t=2552s)
    - [Stanford](https://www.youtube.com/watch?v=5WoItGTWV54)
- Slides/Presentations
    - [Stanford](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf)
    - [Harvard University](https://harvard-iacs.github.io/2019-CS109B/pages/lecture19/presentation/cs109b_lecture19_VAE.pdf)
    - [University of Columbia](https://www.cs.ubc.ca/~lsigal/532S_2018W2/Lecture17.pdf)
    
