# Ghibli Style Transfer using CycleGAN

This project implements a Cycle Generative Adversarial Network (CycleGAN) to perform artistic style transfer, specifically transforming regular images into the distinctive style of Studio Ghibli animations. The model is trained to learn the mapping between two image domains (e.g., real-world photos and Ghibli-style artwork) without requiring paired examples.

# Table of Contents

  - [Features](https://www.google.com/search?q=%23features)
  - [Live Demo](https://www.google.com/search?q=%23live-demo)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Model Architecture](https://www.google.com/search?q=%23model-architecture)
  - [Loss Functions](https://www.google.com/search?q=%23loss-functions)
  - [Getting Started](https://www.google.com/search?q=%23getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
      - [Dataset](https://www.google.com/search?q=%23dataset)
      - [Training](https://www.google.com/search?q=%23training)
      - [Testing and Evaluation](https://www.google.com/search?q=%23testing-and-evaluation)
  - [Results](https://www.google.com/search?q=%23results)
  - [Resources](https://www.google.com/search?q=%23resources)
  - [Contributors](https://www.google.com/search?q=%23contributors)

## Features

  * **Unpaired Image-to-Image Translation**: Utilizes the CycleGAN architecture for style transfer without the need for perfectly matched image pairs.
  * **PyTorch Implementation**: Built using the PyTorch deep learning framework for efficient computation.
  * **Generative Adversarial Networks (GANs)**: Employs a system of generator and discriminator networks to learn the stylistic transformations.
  * **Cycle Consistency Loss**: Incorporates a cycle consistency loss to ensure that the learned mappings are invertible, preserving content across translations.
  * **Streamlit Deployment**: The trained model is deployed as an interactive web application for easy demonstration.

# Live Demo

Experience the Ghibli style transfer in action through our Streamlit application:
[CycleGAN Streamlit App](https://cyclegan-app-6cvc3wgympvy9tskshngmf.streamlit.app/)

# Project Structure

The core logic of the project is contained within the `FiNAL_CV_GHIBHLI_STYLE_TRANSFER.ipynb` Jupyter notebook. This notebook covers:

  * Library installations and imports.
  * Google Drive mounting for dataset access.
  * Dataset loading and preprocessing.
  * Definition of the Generator and Discriminator models.
  * Model initialization, optimizers, and loss functions.
  * The training loop, including checkpointing and image sampling.
  * Sections for testing and evaluation of the trained models.

# Model Architecture

The project is based on the **CycleGAN** architecture, which consists of two generative models and two discriminative models.

  * **Generators (G\_AB and G\_BA)**:

      * Each generator is responsible for translating images from one domain to another (e.g., G\_AB translates from Domain A to Domain B, and G\_BA translates from Domain B to Domain A).
      * The `Generator` class in this implementation includes:
          * Initial convolutional layers for feature extraction.
          * **6 Residual Blocks**: These blocks help in learning identity mappings and facilitate deeper network architectures.
          * Transposed convolutional layers for upsampling and generating the output image.
          * `ReflectionPad2d`, `InstanceNorm2d`, `ReLU`, and `Tanh` activation functions are used throughout the network.

  * **Discriminators (D\_A and D\_B)**:

      * Each discriminator tries to distinguish between real images from a domain and fake (generated) images for that domain.
      * The `Discriminator` class implements convolutional layers with `InstanceNorm2d` and `LeakyReLU` activation functions.

# Loss Functions

The training of the CycleGAN model involves several loss components to ensure effective style transfer and content preservation:

  * **GAN Loss (`nn.MSELoss`)**: This loss is applied to the output of the discriminators to encourage the generators to produce realistic images that can fool the discriminators.
  * **Cycle Consistency Loss (`nn.L1Loss`)**: This crucial loss ensures that if an image is translated from one domain to another and then back to the original domain, the reconstructed image should be identical to the original. This helps in preserving the content of the image during style transfer. There are two such losses: `loss_cycle_A` and `loss_cycle_B`.
  * **Identity Loss (`nn.L1Loss`)**: This loss encourages the generator to preserve the color composition of the input image when translating it to the target domain, particularly when the input image already belongs to the target domain. This helps in preventing unnecessary color changes. There are two such losses: `loss_identity_A` and `loss_identity_B`.

# Getting Started

## Prerequisites

  * Python 3.x
  * PyTorch
  * Torchvision
  * Pillow
  * Numpy
  * Matplotlib
  * tqdm

These libraries are installed via `pip` commands in the notebook.

## Installation

1.  **Clone the GitHub repository** (once it's set up).
2.  **Download the `FiNAL_CV_GHIBHLI_STYLE_TRANSFER.ipynb` notebook** to your local machine or Google Drive.
3.  **Open the notebook** in a Jupyter environment (e.g., Google Colab).

## Dataset

The dataset used here is taken from Keggle.link:- (https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images)


## Training

The training process involves iterating over epochs and updating the Generator and Discriminator networks.

  * **Optimizer**: Adam optimizer is used for both generators and discriminators.
  * **Epochs**: The notebook is configured to train for 50 epochs.
  * **Checkpointing**: The model saves checkpoints periodically (`cyclegan_latest.pth` and epoch-specific checkpoints like `cyclegan_epoch_50.pth`) to allow resuming training or loading trained models for inference.
  * **Progress Tracking**: `tqdm` is used to display training progress.

To start training, run the cells in the "Training Block" section of the notebook.

## Testing and Evaluation

After training, you can test the model on new images:

1.  Ensure the trained models (checkpoints) are loaded. The notebook specifically loads `cyclegan_epoch_50.pth`.
2.  Prepare test images in a specified test folder (e.g., `/content/drive/MyDrive/CV_PROJ_DATA/dataset/testA`).
3.  Run the `test_single_image` function or the loop provided in the "FOR TESTING AND EVALUATION" section to translate images and save the outputs.
4.  The notebook also includes code to visualize original and translated image pairs using `matplotlib`.

# Results


<img width="776" height="260" alt="10" src="https://github.com/user-attachments/assets/ab665b1d-8d6d-4e76-8e7a-031917acf23f" />
<img width="776" height="260" alt="27" src="https://github.com/user-attachments/assets/f71d1631-01cc-4858-8c47-b996d8958866" />
<img width="776" height="260" alt="47" src="https://github.com/user-attachments/assets/83f695cf-df11-4fc2-b753-d8ecf0e54541" />


# Resources

  * **Official CycleGAN Paper**: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
  * **Original PyTorch CycleGAN GitHub**: [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  * **Dataset**: [Real to Ghibli Image Dataset](https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images)

# Contributors

  * Partha Mete
  * Kanan Pandit
