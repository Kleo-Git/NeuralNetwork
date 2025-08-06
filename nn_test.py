import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from losses import Loss, Loss_BinaryCrossEntropy
from layers import Layer_Dense, Layer_Dropout
from activations import Activation_ReLU, Activation_Sigmoid
from combined import Activation_Softmax_Loss_CategoricalCrossEntropy
from optimizers import Optimizer_SGD, Optimizer_Adagrad, Optimizer_RMSprop, Optimizer_Adam

#Initialize nnfs, sets random seed and default float precision
nnfs.init()

#Generate a spiral dataset with 100 samples per class
X, y = spiral_data(samples=100, classes=2)

#Display the spiral data, with a different colour per class
plt.scatter(X[:, 0],X[:, 1], c=y, cmap='brg')
plt.show()

y = y.reshape(-1,1)

#Create a dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

#Create a ReLU activation to be used
activation1 = Activation_ReLU()

#Create a second dense layer with 64 (need 64 since the previous layer has 64 outputs)
#input features and 1 output value
dense2 = Layer_Dense(64,1)

#Create activation sigmoid
activation2 = Activation_Sigmoid()

#Create softmax combined loss and acitvation
loss_function = Loss_BinaryCrossEntropy()

#Create optimizer object
optimizer = Optimizer_Adam(decay_rate=5e-7)


for epoch in range(10001):
    #Perform a forward pass of data through the layer
    dense1.forward(X)
    
    #Forward pass through activation function
    #Takes in output from previous layer
    activation1.forward(dense1.output)
        
    #Forward pass through second dense layer
    #Takes outputs of activation function 1 as inputs
    dense2.forward(activation1.output)
    
    #Forward pass through activation function
    activation2.forward(dense2.output)

    #Perform forward pass through the activation/loss function
    #takes output of second dense layer and returns loss
    data_loss = loss_function.calculate(activation2.output, y)
    
    #Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    
    #Calculate overall loss
    loss = data_loss + regularization_loss
    
    #Calculate the accuracy of the model
    #This is simply how often the models predictions are correct
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f"epoch = {epoch}, " + f"accuracy = {accuracy:.3f}, " + f"loss = {loss:.3f}, " +
              f"regularization_loss = {regularization_loss:.3f}, " +  f"learning rate = {optimizer.current_learning_rate:.3f}")
        
    #Perform backwards pass through the network
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)   
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    #Calculate parameter updates from the optimizer
    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_parameters()

print("accuracy =", accuracy)


# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=2)
#Reshape
y_test = y_test.reshape(-1,1)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through activation function
activation2.forward(dense2.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

























