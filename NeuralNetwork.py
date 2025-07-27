import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

class Activation_ReLU:
    def forward(self, inputs):
        #Applies the ReLU activation function
        #Replaces all negative values with 0, positive values remain unchanged
        self.output = np.maximum(0, inputs)
        
class Activation_Sigmoid:
    def forward(self, inputs):
        #Applies the Sigmoid activation function
        #Returns input values in the range (0,1) using a curve
        #Curve rapidly changes near 0, smoothes out at larger values
        self.output = 1 / (1+np.exp(-inputs))
        
class Activation_Step:
    def forward(self, inputs):
        #Applies the Step activation function
        #Returns 1 for inputs >= 0, otherwise returns 0
        self.output = np.where(inputs >= 0, 1, 0)

# Dense (fully connected) layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #Initialize weights with small random values
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        #Initialize biases with zeroes
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        #Compute the output of a foward pass of a layer
        self.output = np.dot(inputs, self.weights) + self.biases

#Initialize nnfs, sets random seed and default float precision
nnfs.init()

#Generate a spiral dataset with 100 samples per class
X, y = spiral_data(samples=100, classes=3)

#Display the spiral data, with a different colour per class
plt.scatter(X[:, 0],X[:, 1], c=y, cmap='brg')
plt.show()


#Create a dense layer with 2 input features and 3 output neurons
dense1 = Layer_Dense(2, 3)

#Create a ReLU activation to be used
activation1 = Activation_ReLU()

#Perform a forward pass of data through the layer
dense1.forward(X)

#Forward pass through activation function
#Takes in output from previous layer
activation1.forward(dense1.output)

print(dense1.output[:5])
print(activation1.output[:5])



























