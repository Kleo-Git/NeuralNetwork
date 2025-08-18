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


labels = os.listdir("fashion_mnist_images/train")
files = os.listdir("fashion_mnist_images/train/0")

img = cv2.imread("fashion_mnist_images/train/4/0011.png", cv2.IMREAD_UNCHANGED)

plt.imshow(img, cmap = "gray")
plt.show()


