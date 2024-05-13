import numpy as np

class Regression:

    @staticmethod
    def activate(self,Z):
        return np.maximum(Z,0)
    
    @staticmethod
    def der_activate(self,Z):
        return Z > 0
    
    @staticmethod
    def softmax(self,Z):
        return np.exp(Z)/np.sum(np.exp(Z),axis= 0)
    
    