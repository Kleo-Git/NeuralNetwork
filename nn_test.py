import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, sine_data
from losses import Loss, Loss_BinaryCrossEntropy, Loss_MeanSquaredError, Loss_CategoricalCrossEntropy
from layers import Layer_Dense, Layer_Dropout
from activations import Activation_ReLU, Activation_Sigmoid, Activation_Linear, Activation_Softmax
from combined import Activation_Softmax_Loss_CategoricalCrossEntropy
from optimizers import Optimizer_SGD, Optimizer_Adagrad, Optimizer_RMSprop, Optimizer_Adam
from model import Model
from accuracy import Accuracy_Regression, Accuracy_Categorical
from create_data import create_data_mnist


#Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
