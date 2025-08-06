import numpy as np

class Activation_ReLU:
    def forward(self, inputs):
        
        self.inputs = inputs
        #Applies the ReLU activation function
        #Replaces all negative values with 0, positive values remain unchanged
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        
        #Make a copy so we can modify without changing the original values
        self.dinputs = dvalues.copy()
        
        #Zero gradient if inputs are less then 0
        self.dinputs[self.inputs <= 0] = 0
        
class Activation_Sigmoid:
    def forward(self, inputs):
        
        #Applies the Sigmoid activation function
        #Returns input values in the range (0,1)
        #Curve rapidly changes near 0, smoothes out at larger values
        self.inputs = inputs
        self.output = 1 / (1+np.exp(-inputs))
    
    def backward(self, dvalues):
        #Gradient of sigmoid function
        self.dinputs = dvalues * self.output * (1-self.output)
        
class Activation_Step:
    def forward(self, inputs):
        
        #Applies the Step activation function
        #Returns 1 for inputs >= 0, otherwise returns 0
        self.output = np.where(inputs >= 0, 1, 0)
    
    def backward(self, dvalues):
        #Derivative is 0 at all points, except 0 where it is undefined
        #This is why we dont use the step function
        self.dinputs = np.zeros_like(dvalues)
        
class Activation_Linear:
    #Forward pass
    def forward(self, inputs):
        #Remember values since simple linear function e.g. y=x
        self.inputs = inputs
        self.output = inputs
        
    def backward(self, dvalues):
        #Derivative is simply 1*dvalues
        self.dinputs = dvalues.copy()
        
        
class Activation_Softmax:
    def forward(self, inputs): 
        
        self.inputs = inputs
        #Calculate the unormalized probabilties
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        #Normalize the probabilities for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
    def backward(self, dvalues):
        
        #Create unintialized array with shape of dvalues
        self.dinputs = np.empty_like(dvalues)
        
        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1,1)
            #Calculate the jacobian
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample wise gradient and add to array of sample gradients
            self.dinputs[i] = np.dot(jacobian, single_dvalues)