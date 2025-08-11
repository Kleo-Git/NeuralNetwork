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
from accuracy import Accuracy_Regression, Accuracy_Categorical

X, y = spiral_data(100,2)
X_test, y_test = spiral_data(100,2)


y = y.reshape(-1,1)
y_test = y_test.reshape(-1,1)

model = Model()

model.add(Layer_Dense(2, 64, weight_regularizer_l1=5e-4, weight_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Sigmoid())

model.set(loss = Loss_BinaryCrossEntropy(),
          optimizer=Optimizer_Adam(decay_rate=5e-7),
          accuracy=Accuracy_Categorical(binary=True)
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)










