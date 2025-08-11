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

X, y = spiral_data(1000,3)
X_test, y_test = spiral_data(1000,3)

model = Model()

model.add(Layer_Dense(2, 256, weight_regularizer_l1=5e-4, weight_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(256, 3))
model.add(Activation_Softmax())

model.set(loss = Loss_CategoricalCrossEntropy(),
          optimizer=Optimizer_Adam(learning_rate=0.05, decay_rate=5e-5),
          accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=1000, print_every=100)










