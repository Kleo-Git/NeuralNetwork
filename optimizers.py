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
        

class Optimizer_Adagrad:
    #Initialize optimizer
    #Set learning rate = 1 by default
    def __init__ (self, learning_rate=1.0, decay_rate=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.epsilon = epsilon
    
    #Call once before any parameter updates
    def pre_update_parameters(self):
        if self.decay_rate:
            #Allow for learning rate to decrease over iterations
            #Allows for fast learning intially to prevent being trapped in local minima
            self.current_learning_rate = self.learning_rate * (1 / (1+ self.decay_rate * self.iterations))
    
    #Update parameters based on learning rate
    def update_parameters(self, layer):
        
        #Create cache arrays if they dont exist already
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        #Update cache with squared graidents
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        
        #Find weight/bias updates based on square rooted cache
        layer.weights += - self.current_learning_rate * layer.dweights \
            / (np.sqrt(layer.weight_cache) + self.epsilon)

        layer.biases += - self.current_learning_rate * layer.dbiases \
            / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    #Call once after parameter updates
    def post_update_parameters(self):
        #Count iterations
        self.iterations += 1        
        
class Optimizer_RMSprop:
    #Initialize optimizer
    #Set learning rate = 0.001 by default due to this optimizer carries over far more 
    #momentum of gradient
    def __init__ (self, learning_rate=0.001, decay_rate=0.0, epsilon=1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    #Call once before any parameter updates
    def pre_update_parameters(self):
        if self.decay_rate:
            #Allow for learning rate to decrease over iterations
            #Allows for fast learning intially to prevent being trapped in local minima
            self.current_learning_rate = self.learning_rate * (1 / (1+ self.decay_rate * self.iterations))
    
    #Update parameters based on learning rate
    def update_parameters(self, layer):
        #Create cache arrays if they dont exist already
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        #Update cache with squared graidents
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho)*layer.dbiases**2
        
        #Find weight/bias updates based on square rooted cache
        layer.weights += - self.current_learning_rate * layer.dweights \
            / (np.sqrt(layer.weight_cache) + self.epsilon)
    
        layer.biases += - self.current_learning_rate * layer.dbiases \
            / (np.sqrt(layer.bias_cache) + self.epsilon)
    
    #Call once after parameter updates
    def post_update_parameters(self):
        #Count iterations
        self.iterations += 1    

class Optimizer_Adam:
    #Initialize optimizer
    #Set learning rate = 1 by default
    def __init__ (self, learning_rate=0.001, decay_rate=0.0, epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    
    #Call once before any parameter updates
    def pre_update_parameters(self):
        if self.decay_rate:
            #Allow for learning rate to decrease over iterations
            #Allows for fast learning intially to prevent being trapped in local minima
            self.current_learning_rate = self.learning_rate * (1 / (1+ self.decay_rate * self.iterations))
    
    #Update parameters based on learning rate
    def update_parameters(self, layer):
        #Create momentum arrays if they dont exist already
        if not hasattr(layer, "weight_momentums"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        #Update momentum with current gradients
        layer.weight_momentums = self.beta1* layer.weight_momentums + (1-self.beta1) *layer.dweights
        layer.bias_momentums = self.beta1* layer.bias_momentums + (1-self.beta1) *layer.dbiases
        
        #Find the corrected momentum
        weight_momentums_corrected = layer.weight_momentums/(1-self.beta1 ** (self.iterations+1))
        bias_momentums_corrected = layer.bias_momentums/(1-self.beta1 ** (self.iterations+1))
        
        #Update cache with squared current gradients
        layer.weight_cache = self.beta2 * layer.weight_cache + (1-self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1-self.beta2) * layer.dbiases**2
        
        #Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    #Call once after parameter updates
    def post_update_parameters(self):
        #Count iterations
        self.iterations += 1
        
