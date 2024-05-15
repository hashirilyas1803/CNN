import numpy as np

class Regression:

    @staticmethod
    def sigmoid(Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def activate(Z):
        return np.maximum(Z,0)
    
    @staticmethod
    def der_activate(Z):
        return Z > 0
    
    @staticmethod
    def softmax(Z):
        return np.exp(Z)/np.sum(np.exp(Z),axis= 0)
    
    