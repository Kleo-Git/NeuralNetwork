import numpy as np

class Optimizer_SGD:
    #Initialize optimizer
    #Set learning rate = 1 by default
    def __init__ (self, learning_rate=1.0, decay_rate=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
    
    #Call once before any parameter updates
    def pre_update_parameters(self):
        if self.decay_rate:
            #Allow for learning rate to decrease over iterations
            #Allows for fast learning intially to prevent being trapped in local minima
            self.current_learning_rate = self.learning_rate * (1 / (1+ self.decay_rate * self.iterations))
    
    #Update parameters based on learning rate
    def update_parameters(self, layer):
        #Update weights and biases by some value multiplied by their gradients
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases
    
    #Call once after parameter updates
    def post_update_parameters(self):
        #Count iterations
        self.iterations += 1