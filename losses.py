import numpy as np

class Loss:
    #Calculates the data and regularization losses
    def calculate(self, output, y):
        
        #Calculates the sample losses
        sample_losses = self.forward(output, y)
        
        #Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        #Return mean loss
        return data_loss, self.regularization_loss()
    
    def regularization_loss(self, layer):
        #Start at 0 by default
        regularization_loss = 0
        
        #Calculate regularization loss, over all trainable layers
        for layer in self.trainable_layers:
            #Perform L1 regularization - weights
            #Prevent performing needless caluclations when factor = 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            #Perform L2 regularization - weights    
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights*layer.weights)
            #Perform L1 regularization - biases       
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            #Perform L2 regularization - biases  
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
                
        return regularization_loss
    
    #Set and remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
            
        
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
        
class Loss_BinaryCrossEntropy(Loss):
    
    #Forward Pass
    def forward(self, y_predicted, y_actual):
        
        #Clip data to prevent division by 0
        #Clip data on both sides to not drag the mean towards a value
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
        
        #Calculate losses sample wise
        sample_losses = -(y_actual * np.log(y_predicted_clipped) + (1-y_actual) \
                          * np.log(1 - y_predicted_clipped))
        sample_losses = np.mean(sample_losses, axis = -1)
        
        #Return losses
        return sample_losses
        
    def backward(self, dvalues, y_actual):
        
        #No. samples
        samples = len(dvalues)
        #No. outputs in every sample
        outputs = len(dvalues[0])
        
        #Clip data for same reasons as forward pass clipping
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        #Calculate gradient
        self.dinputs = -(y_actual / clipped_dvalues - (1-y_actual) \
                         / (1 - clipped_dvalues)) / outputs
        
        #Normalize gradient
        self.dinputs = self.dinputs/samples
        
class Loss_MeanSquaredError(Loss):
    
    def forward(self, y_predicted, y_actual):
        #Basic loss calculation for the sample
        sample_losses = np.mean((y_actual - y_predicted)**2, axis=-1)
        
        return sample_losses
    
    def backward(self, dvalues, y_actual):
        
        #No. samples
        samples = len(dvalues)
        #No. outputs in every sample
        outputs = len(dvalues[0])
        
        #Gradient on values based on MSE equation
        self.dinputs = -2 * (y_actual - dvalues) / outputs
        #Normalize gradient
        self.dinputs = self.dinputs / samples
        
class Loss_MeanAbsoluteError(Loss):
    
    def forward(self, y_predicted, y_actual):
        #Basic loss calculation for the sample
        sample_losses = np.mean(np.abs(y_actual - y_predicted), axis=-1)

        return sample_losses

    def backward(self, dvalues, y_actual):
        
        #No. samples
        samples = len(dvalues)
        #No. outputs in every sample
        outputs = len(dvalues[0])
        
        #Gradient on values based on MAE equation
        self.dinputs = np.sign(y_actual - dvalues) / outputs
        #Normalize gradient
        self.dinputs = self.dinputs / samples
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

