import os
import cv2
import numpy as np

def load_mnist_dataset(dataset, path):
    #Scan all directories and create list of labels    
    labels = os.listdir(os.path.join(path, dataset))
    
    #Lists for samples and labels
    X=[];y=[]
    
    #For each label folder
    for label in labels:
        #And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            
            #Append it and a label to our lists
            X.append(image)
            y.append(label)
    
    #Convert to numpy arrays and return
    return np.array(X), np.array(y).astype("uint8")
    
def create_data_mnist(path):
    
    #Load both sets seperately
    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)

    #Return all data
    return X, y, X_test, y_test

if __name__ == "__main__":
    #Create dataset
    X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')