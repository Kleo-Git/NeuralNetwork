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
        
class Activation_Softmax:
    def forward(self, inputs): 
        
        #Calculate the unormalized probabilties
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        #Normalize the probabilities for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
class Loss:
    #Calculates the data and regularization losses
    def calculate(self, output, y):
        
        #Calculates the sample losses
        sample_losses = self.forward(output, y)
        
        #Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        #Return mean loss
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    
    #Forward Pass
    def forward(self, y_predicted, y_actual):
        
        #Number of samples in a batch
        samples = len(y_predicted)
        
        #Clip data to prevent division by 0, and avoid shifting mean towards a value
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
        
        #Probabilities for target values
        #If labels are sparse
        if len(y_actual.shape) == 1:
            correct_confidences = y_predicted_clipped[range(samples), y_actual]
            
        #If labels are one-hot encoded
        elif len(y_actual.shape) == 2:
            correct_confidences = np.sum(y_predicted_clipped * y_actual, axis = 1)
            
        #Calculate the loss value
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
            

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

#Create a second dense layer with 3 (need 3 since the previous layer has 3 outputs)
#input features and 3 output values
dense2 = Layer_Dense(3,3)

#Create a softmax activation
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()

#Perform a forward pass of data through the layer
dense1.forward(X)

#Forward pass through activation function
#Takes in output from previous layer
activation1.forward(dense1.output)

#Forward pass through second dense layer
#Takes outputs of activation function 1 as inputs
dense2.forward(activation1.output)

#Make a forward pass through the softmax activation function
activation2.forward(dense2.output)

print(activation2.output[:5])

#Forward pass through activation function
#Takes output of second dense layer and returns loss
loss = loss_function.calculate(activation2.output, y)

print('loss:', loss)





























