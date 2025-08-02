import numpy as np

class Optimizer_SGD:
    #Initialize optimizer
    #Set learning rate = 1 by default
    def __init__ (self, learning_rate=1.0, decay_rate=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.momentum = momentum
    
    #Call once before any parameter updates
    def pre_update_parameters(self):
        if self.decay_rate:
            #Allow for learning rate to decrease over iterations
            #Allows for fast learning intially to prevent being trapped in local minima
            self.current_learning_rate = self.learning_rate * (1 / (1+ self.decay_rate * self.iterations))
    
    #Update parameters based on learning rate
    def update_parameters(self, layer):
        #If using momentum
        if self.momentum:
            #Create momentum arrays if they dont exist already
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            #Find weight updates with momentum, use previous updates by some retention factor
            weight_updates = self.momentum * layer.weight_momentums \
                - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums \
                - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            #If not using momentum calculate adjustments with
            #simple learning rate * gradient calculations.
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            
            
        #Update weights and biases the calculated update amount
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    #Call once after parameter updates
    def post_update_parameters(self):
        #Count iterations
        self.iterations += 1