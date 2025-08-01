import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

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
        #Returns input values in the range (0,1) using a curve
        #Curve rapidly changes near 0, smoothes out at larger values
        self.output = 1 / (1+np.exp(-inputs))
    
    def backward(self, dvalues):
        
        #Make a copy so we can modify without changing the original values
        self.dinputs = dvalues.copy()
        
        #Zero gradient if inputs are less then 0
        self.dinputs = dvalues * (self.output * (1-self.output))
        
class Activation_Step:
    def forward(self, inputs):
        
        #Applies the Step activation function
        #Returns 1 for inputs >= 0, otherwise returns 0
        self.output = np.where(inputs >= 0, 1, 0)
    
    def backward(self, dvalues):
        #Derivative is 0 at all points, except 0 where it is undefined
        #This is why we dont use the step function
        self.dinputs = np.zeros_like(dvalues)
        
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
    
    def backward(self, dvalues, y_actual):
        
        #Number of samples
        samples = len(dvalues)
        #Number of labels in every sample
        labels = len(dvalues[0])
        
        #If labels are sparse turn them into one-hot vector
        if len(y_actual.shape) == 1:
            y_actual = np.eye(labels)[y_actual]
            
        #Calculate gradient based on derivative function
        self.dinputs = -y_actual/dvalues
        #Normalize gradient
        self.dinputs = self.dinputs/samples

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

#Create softmax combined loss and acitvation
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

#Perform a forward pass of data through the layer
dense1.forward(X)

#Forward pass through activation function
#Takes in output from previous layer
activation1.forward(dense1.output)

#Forward pass through second dense layer
#Takes outputs of activation function 1 as inputs
dense2.forward(activation1.output)

#Perform forward pass through the activation/loss function
#takes output of second dense layer and returns loss
loss = loss_activation.forward(dense2.output, y)

print(loss_activation.output[:5])

print("loss =", loss)

#Calculate the accuracy of the model
#This is simply how often the models predictions are correct
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print("accuracy =", accuracy)

#Perform a backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)





























