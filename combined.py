import numpy as np
from activations import Activation_Softmax
from losses import Loss_CategoricalCrossEntropy

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    #Initialize activation and loss objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
        
    def forward(self, inputs, y_actual):
            #Output layer's activation function
            self.activation.forward(inputs)
            #Set output
            self.output = self.activation.output
            #Calculate loss value
            return self.loss.calculate(self.output, y_actual)
    
    def backward(self, dvalues, y_actual):
        #Number of samples
        samples = len(dvalues)
        
        #If labels are one-hot vectors turn them to discrete values
        if len(y_actual.shape) == 2:
            y_actual = np.argmax(y_actual, axis=1)
            
        #Copy for future modification
        self.dinputs = dvalues.copy()
        #Calculate gradient
        self.dinputs[range(samples), y_actual] -= 1
        #Normalize
        self.dinputs = self.dinputs / samples