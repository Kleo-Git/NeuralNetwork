import numpy as np

# Dense (fully connected) layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        
        #Initialize weights with small random values
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        
        #Initialize biases with zeroes
        self.biases = np.zeros((1, n_neurons))
        
        #Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs, training):
        
        self.inputs = inputs
        #Compute the output of a foward pass of a layer
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        
        #Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        #Gradients on regularization
        #L1 gradient - weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        #L2 gradient - weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        #L1 gradient - biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.biases_regularizer_l1 * dL1
            
        #L2 gradient - biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        #Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Layer_Dropout:
    
    #Initialize
    def __init__(self, rate):
        #Store the rate, inverted since if we want a
        #dropout of 0.1, we need success rate of 0.9
        self.rate = 1-rate
        
    #Forward pass
    def forward(self, inputs, training):
        #Store input values
        self.inputs = inputs
        
        #If not in training mode, return values
        if not training:
            self.output = inputs.copy()
            return
        
        #Generate and save the scaled mask
        #Scaled to not not affect sum total given dropout
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        #Apply mask to output values
        self.output = inputs * self.binary_mask
    
    #Backward pass
    def backward(self, dvalues):
        #Gradient on values
        self.dinputs = dvalues * self.binary_mask
        
        
class Layer_Input:
    #Forward pass
    def forward(self, inputs, training):
        #Used for the model, the dataset is considered the first 'layer'
        self.output = inputs
    
    