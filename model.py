import numpy as np
from layers import Layer_Input
from activations import Activation_Softmax
from losses import Loss_CategoricalCrossEntropy
from combined import Activation_Softmax_Loss_CategoricalCrossEntropy

class Model:
    
    def __init__(self):
        #Create list of network objects
        self.layers=[]
        #Softmax classifier's output object
        self.softmax_classifier_output = None
        
    #Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
        
    #Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    #Finalize the model
    def finalize(self):
        
        #Create and set input layer
        self.input_layer = Layer_Input()
        
        #Count all the objects
        layer_count = len(self.layers)
        
        #Initialize a list of all trainable layers
        self.trainable_layers = []
        
        #Iterate the objects
        for i in range(layer_count):
            
            #First layer, has input layer as the previous object
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            #All layers except first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
                
            #The last layer, the next object is the loss
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            #If a layer has a weights attribute, it must be a trainable layer
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
            
        #Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)
        
        #Creates an object of combined activation and loss functions
        #If the output function is softmax and loss function is cross-entropy
        if isinstance(self.layers[-1], Activation_Softmax) and \
            isinstance(self.loss, Loss_CategoricalCrossEntropy):
                self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()
                
    #Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data = None):
        
        #Initialize accuracy object
        self.accuracy.init(y)
        
        #Main training loop
        for epoch in range(1,epochs+1):
            #Forward pass
            output = self.forward(X, training = True)
            
            #Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization = True)
            loss = data_loss + regularization_loss
            
            #Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)    
            
            #Perform backward pass
            self.backward(output, y)
            
            #Optimize the network
            self.optimizer.pre_update_parameters()
            for layer in self.trainable_layers:
                self.optimizer.update_parameters(layer)
            self.optimizer.post_update_parameters()
            
            if not epoch % print_every:
                print(f"epoch = {epoch}, " + f"accuracy = {accuracy:.3f}, " + f"loss = {loss:.3f}, " + f"data_loss = {data_loss:.3f}, " +
                      f"regularization_loss = {regularization_loss:.3f}, " +  f"learning rate = {self.optimizer.current_learning_rate:.8f}")
            
        if validation_data is not None:
            
            #Defining the validation data
            X_val, y_val = validation_data
            
            #Perform forward pass
            output = self.forward(X_val, training = False)
            
            #Calculate loss
            loss = self.loss.calculate(output, y_val)
            
            #Get predictions and calcualte accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            
            print("validation " + f"accuracy = {accuracy:.3f}, " 
                  + f"loss = {loss:.3f}, ") 
        
    #Perform forward pass
    def forward(self, X, training):
        #Calls the input layer, setting the output property correctly
        self.input_layer.forward(X, training)
        
        #Call forward method for every object, pass output as input of next layer
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        #Last object in list, return its value
        return layer.output
    
    #Performs backward pass
    def backward(self, output, y):
        
        #If softmax classifier
        if self.softmax_classifier_output is not None:
            #Call backward method
            self.softmax_classifier_output.backward(output, y)
            
            #We won't call backward method on last layer
            #So set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            #Call backward method going through all objects but the last reversed
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                
            return
        
        #Call backward method on the loss
        self.loss.backward(output, y)
        
        #Call backward method going through all objects reversed
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
        