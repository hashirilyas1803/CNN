# CNN Library

**Authors:** Muhammad Hashir Ilyas (26972), Muhammad Imad Raza (26953), Saad Imam (27079)  
**Institute:** Institute of Business Administration (IBA), Karachi  

This repository contains a modular Convolutional Neural Network (CNN) built entirely from scratch using Python and `NumPy`. This project was developed without any high-level machine learning frameworks (e.g., TensorFlow, PyTorch) to gain a fundamental understanding of forward propagation, calculus-based backpropagation, and tensor manipulation.

## Core Framework Features

The framework relies on an object-oriented structure where each layer handles its own mathematical transformations and gradient computations:

* **Vectorized Convolution Layer (`Convolution_Layer.py`):** 
  * Avoids slow nested `for` loops by utilizing `numpy.lib.stride_tricks.as_strided` to manipulate memory strides.
  * Flattens sliding-window patches to execute 2D cross-correlation as a highly optimized, vectorized matrix dot-product.
  * Computes exact gradients w.r.t filters and inputs for the backward pass.
* **Max Pooling Layer (`Pooling.py`):** 
  * Implements spatial downsampling with configurable strides.
  * Tracks maximum indices during the forward pass to correctly route gradients during backpropagation.
* **Fully Connected Layer (`Fully_Connected.py`):** 
  * Standard Multi-Layer Perceptron (MLP) operations applying chain-rule backpropagation to update weights and biases based on learning rates.
* **Activations (`Regression.py`):** 
  * Includes manual implementations of ReLU, Sigmoid, Softmax, and their respective derivatives.

## Proof of Concept: Signature Forgery Detection

To validate the mathematical correctness of the library, we wrote a driver script (`CNN.py`) to process and classify a dataset of real and forged signatures.
* Images are loaded, resized to 64x64, converted to grayscale, and normalized.
* The driver initializes a basic architecture (1 Conv -> 1 Pool -> 1 FC) and executes custom epoch loops.
* *Note: As this project prioritizes understanding the underlying linear algebra over achieving state-of-the-art accuracy, the training loop utilizes a naive hyperparameter configuration and basic delta-rule loss rather than a production-scale optimization strategy.*

## Web API Deployment (`server.py`)

The inference logic is wrapped in a lightweight `Flask` backend to demonstrate how a custom model can be served to an end-user application. 
* Provides a web interface to upload signature images.
* Routes images through the trained NumPy layers.
* Returns a binary classification regarding the authenticity of the signature.

## Setup and Usage

1. Clone the repository and install dependencies (`numpy`, `Pillow`, `flask`).
2. Run `CNN.py` to initiate the training/testing loop and save weights to `weights.npz`.
3. Run `server.py` to launch the Flask web application on `localhost:4448`.