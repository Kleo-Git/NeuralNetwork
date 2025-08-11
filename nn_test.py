import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, sine_data
from losses import Loss, Loss_BinaryCrossEntropy, Loss_MeanSquaredError
from layers import Layer_Dense, Layer_Dropout
from activations import Activation_ReLU, Activation_Sigmoid, Activation_Linear
from combined import Activation_Softmax_Loss_CategoricalCrossEntropy
from optimizers import Optimizer_SGD, Optimizer_Adagrad, Optimizer_RMSprop, Optimizer_Adam
from model import Model
from accuracy import Accuracy_Regression

X, y = sine_data()

model = Model()

model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

model.set(loss = Loss_MeanSquaredError(),
          optimizer=Optimizer_Adam(learning_rate=0.005, decay_rate=1e-3),
          accuracy=Accuracy_Regression()
)

model.finalize()

model.train(X, y, epochs=10000, print_every=100)










