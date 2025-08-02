import numpy as np

# Dense (fully connected) layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        
        #Initialize weights with small random values
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        
        #Initialize biases with zeroes
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        
        self.inputs = inputs
        #Compute the output of a foward pass of a layer
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        #Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)