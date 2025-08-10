import numpy as np
from layers import Layer_Input


class Model:
    
    def __init__(self):
        #Create list of network objects
        self.layers=[]
        
    #Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
        
    #Set loss and optimizer
    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

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
                
            #If a layer has a weights attribute, it must be a trainable layer
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
                
    #Train the model
    def train(self, X, y, *, epochs=1, print_every=1):
        #Main training loop
        for epoch in range(1,epochs+1):
            #Forward pass
            output = self.forward(X)
            
            #Temp
            print(output)
            exit()
    
    #Perform forward pass
    def forward(self, X):
        #Calls the input layer, setting the output property correctly
        self.input_layer.forward(X)
        
        #Call forward method for every object, pass output as input of next layer
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        #Last object in list, return its value
        return layer.output
    
            
    
        
        