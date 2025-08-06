import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data, sine_data
from losses import Loss, Loss_BinaryCrossEntropy, Loss_MeanSquaredError
from layers import Layer_Dense, Layer_Dropout
from activations import Activation_ReLU, Activation_Sigmoid, Activation_Linear
from combined import Activation_Softmax_Loss_CategoricalCrossEntropy
from optimizers import Optimizer_SGD, Optimizer_Adagrad, Optimizer_RMSprop, Optimizer_Adam

#Initialize nnfs, sets random seed and default float precision
nnfs.init()

#Generate a spiral dataset with 100 samples per class
#X, y = spiral_data(samples=100, classes=2)

#Use a sine function to generate date for regression model
X, y = sine_data()

#Display the spiral data, with a different colour per class
#plt.scatter(X[:, 0],X[:, 1], c=y, cmap='brg')

#Display sine data
plt.plot(X,y)
plt.show()

#Create a dense layer with 1 input feature and 64 output values
dense1 = Layer_Dense(1, 64)

#Create a ReLU activation to be used
activation1 = Activation_ReLU()

#Create a second hiden layer, necessary for non linear functions
dense2 = Layer_Dense(64,64)

#Create activation relu
activation2 = Activation_ReLU()

#Create a third dense layer with 64 (need 64 since the previous layer has 64 outputs)
#input features and 1 output value
dense3 = Layer_Dense(64, 1)

#Create linear activation
activation3 = Activation_Linear()

#Create mean squared error loss function
loss_function = Loss_MeanSquaredError()

#Create optimizer object
optimizer = Optimizer_Adam(learning_rate=0.005, decay_rate=1e-3)

#Use an accuracy precision to check the accuracy of the model
#Since outputting continous numbers, not so simple to check if they exactly match
accuracy_precision = np.std(y) / 250

#Train within loop
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
    
    #Forward pass through third dense layer
    dense3.forward(activation2.output)
    
    #Forward pass through third activation function
    activation3.forward(dense3.output)
    
    #Perform forward pass through the activation/loss function
    #takes output of second dense layer and returns loss
    data_loss = loss_function.calculate(activation3.output, y)
    
    #Calculate regularization penalty
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) \
                            + loss_function.regularization_loss(dense3)
    
    #Calculate overall loss
    loss = data_loss + regularization_loss
    
    #Calculate the accuracy of the model
    #This is simply how often the models predictions are correct
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    
    if not epoch % 100:
        print(f"epoch = {epoch}, " + f"accuracy = {accuracy:.3f}, " + f"loss = {loss:.3f}, " +
              f"regularization_loss = {regularization_loss:.3f}, " +  f"learning rate = {optimizer.current_learning_rate:.3f}")
        
    #Perform backwards pass through the network
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)   
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)    
    
    #Calculate parameter updates from the optimizer
    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)
    optimizer.update_parameters(dense3)
    optimizer.post_update_parameters()

print("accuracy =", accuracy)


# Create test dataset
#X_test, y_test = spiral_data(samples=100, classes=2)

X_test, y_test = sine_data()

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

dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()



    



















