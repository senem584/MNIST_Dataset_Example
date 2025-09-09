# MNIST_Dataset_Example
This project demostrates the MNIST dataset which contains a large collection of handwritten digits used to train a neural network while utilizing TensorFlow + Keras. The goal is for demostration of image classification. 
# Table Of Contents
- [Implementation](#implementation)
- [Requirements](#requirments)
- [Error Handling](#error-handling)
- [References](#references)
# Implementation
The models implementation consists of an input of the MNIST dataset in which it involves a collection of 70,000 handwritten digits (0-9), with each image being 28x28 pixels. The model is trained using 25 EPOCHs in which it is passed through 25 times, with a validation split of 0.2 to prevent overfitting and validates on a specific portion of the data. 
# Requirments 
This project requires tensorflow, keras, and scikit-learn. It was developed using Python 3.12

Use 'pip install -r requirements.txt' to install the following dependencies:

```
tensorflow==2.20.0
keras==3.11.3
scikit-learn==1.7.1
```
# Error Handling 
This project does not have any error handling.
# References 
- [1]GeeksforGeeks, “MNIST Dataset : Practical Applications Using Keras and PyTorch,” GeeksforGeeks, May 2024. https://www.geeksforgeeks.org/machine-learning/mnist-dataset/
‌
