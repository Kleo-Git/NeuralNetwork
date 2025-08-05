import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from losses import Loss
from layers import Layer_Dense, Layer_Dropout
from activations import Activation_ReLU
from combined import Activation_Softmax_Loss_CategoricalCrossEntropy
from optimizers import Optimizer_SGD, Optimizer_Adagrad, Optimizer_RMSprop, Optimizer_Adam

#Initialize nnfs, sets random seed and default float precision
nnfs.init()

#Generate a spiral dataset with 100 samples per class
X, y = spiral_data(samples=100, classes=3)

#Display the spiral data, with a different colour per class
plt.scatter(X[:, 0],X[:, 1], c=y, cmap='brg')
plt.show()

#Create a dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

#Create a ReLU activation to be used
activation1 = Activation_ReLU()

# Create dropout layer
dropout1 = Layer_Dropout(0.05)

#Create a second dense layer with 64 (need 64 since the previous layer has 64 outputs)
#input features and 3 output values
dense2 = Layer_Dense(64,3)

#Create softmax combined loss and acitvation
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

#Create optimizer object
optimizer = Optimizer_Adam(learning_rate=0.05, decay_rate=5e-5)


for epoch in range(10001):
    #Perform a forward pass of data through the layer
    dense1.forward(X)
    
    #Forward pass through activation function
    #Takes in output from previous layer
    activation1.forward(dense1.output)
    
    # Perform a forward pass through Dropout layer
    dropout1.forward(activation1.output)
        
    #Forward pass through second dense layer
    #Takes outputs of activation function 1 as inputs
    dense2.forward(dropout1.output)

    #Perform forward pass through the activation/loss function
    #takes output of second dense layer and returns loss
    data_loss = loss_activation.forward(dense2.output, y)
    
    #Calculate regularization penalty
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    
    #Calculate overall loss
    loss = data_loss + regularization_loss
    
    #Calculate the accuracy of the model
    #This is simply how often the models predictions are correct
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    
    if not epoch % 100:
        print(f"epoch = {epoch}, " + f"accuracy = {accuracy:.3f}, " + f"loss = {loss:.3f}, " +
              f"regularization_loss = {regularization_loss:.3f}, " +  f"learning rate = {optimizer.current_learning_rate:.3f}")
        
    #Perform backwards pass through the network
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)    
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)
    
    #Calculate parameter updates from the optimizer
    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.post_update_parameters()

print("accuracy =", accuracy)


# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
# Perform a forward pass of our testing data through this layer
dense1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
 y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

























