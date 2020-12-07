# Tests regarding the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset

In this file we present the results of some tests we conducted to assert how the VAE and betaVAE models perform on simple Datasets, like the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist).

The fixed configuration used for the following experiments is:

- Convolutional Layers = 3
- Convolution Channels = \[128, 256, 512\]
- Convolution Kernel Sizes = \[(7, 7), (7, 7), (5, 5)\]
- Convolution Strides = \[(1, 1), (1, 1), (1, 1)\]
- Convolution Paddings = \[(1, 1), (1, 1), (1, 1)\]

- Epochs = 3
- Batch Size = 32
- Learning Rate = 1e-5
- beta = 3
<br> </br>


Let's see what happens when we change the dimension of the latent vector z:

<br> </br>
- Latent Vector Dimension = 1
    
    - VAE

        Sample

        ![image](./images/fmnist_z_1_s.png)

        Reconstruction

        ![image](./images/fmnist_z_1_r.png)

    - B-VAE

        Sample

        ![image](./images/fmnist_z_1_s_B_3.png)

        Reconstruction

        ![image](./images/fmnist_z_1_r_B_3.png)

<br> </br>
- Latent Vector Dimension = 2

    - VAE

        Sample

        ![image](./images/fmnist_z_2_s.png)

        Reconstruction

        ![image](./images/fmnist_z_2_r.png)

    - B-VAE

        Sample

        ![image](./images/fmnist_z_2_s_B_3.png)

        Reconstruction

        ![image](./images/fmnist_z_2_r_B_3.png)

<br> </br>
- Latent Vector Dimension = 8

    - Sample

        ![image](./images/fmnist_z_8_s.png)

        Reconstruction

        ![image](./images/fmnist_z_8_r.png)

    - B-VAE

        Sample

        ![image](./images/fmnist_z_8_s_B_3.png)

        Reconstruction

        ![image](./images/fmnist_z_8_r_B_3.png)
<br> </br>

- Latent Vector Dimension = 16

    - Sample

        ![image](./images/fmnist_z_16_s.png)

        Reconstruction

        ![image](./images/fmnist_z_16_r.png)

    - B-VAE

        Sample

        ![image](./images/fmnist_z_16_s_B_3.png)

        Reconstruction

        ![image](./images/fmnist_z_16_r_B_3.png)
<br> </br>


## Observations

We can see a clear tradeoff for increasing the dimension of the latent vector z:
- Increasing the dimension helps in reconstructing the images more accurately
- Decreasing the dimension helps in better sampling, as higher dimension latent vectors sample sharper images
