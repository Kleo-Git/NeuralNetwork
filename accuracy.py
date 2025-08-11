import numpy as np

#Base accuracy class
class Accuracy:
    #Calculates an accuracy
    def calculate(self, predictions, y):
        
        #Get comparison results
        comparisons = self.compare(predictions, y)
        
        #Calculate an accuracy
        accuracy = np.mean(comparisons)
        
        return accuracy
    

#Accuracy class for regression model
class Accuracy_Regression(Accuracy):
    
    def __init__(self):
        #Create precision property
        self.precision = None
        
    #Calculates precision value based on passed in values
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
        
    #Compare predictions to true values
    def compare(self, predictions, y):
        return np.abs(predictions - y) < self.precision

#Accuracy class for classification model
class Accuracy_Categorical(Accuracy):
    
    def __init__(self, *, binary=False):
        self.binary = binary
        
    #No intialization needed
    def init(self, y):
        pass
    
    #Compare predictions to true values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
        return predictions == y
    

    
    
    
