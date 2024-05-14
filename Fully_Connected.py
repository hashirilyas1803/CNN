import numpy as np
import math 
from Regression import Regression

class fully_connected:

   def __init__(self):

    def forward_prop(self, X, softmax_size):
        self.input = X
        self.weight = np.random.randn(softmax_size, input_size) / input_size
        self.bias = np.zeros((softmax_size,1))
        # Perform the matrix multiplication
        Z = np.matmul(self.weight, X) + self.bias

        # Apply the ReLU activation function
        A = np.maximum(0, Z)

        return A
    
    def backward_prop(self,learning_rate,Y):
    
        dl_dw = np.zeros(self.weight.shape) 
        dl_db = np.zeros(self.bais.shape)
        dl_back = np.zeros(self.input.shape)
        
        for i in range(self.input.shape[2]):
            
            output = np.matmul(self.weight,self.input[:,i]) + self.bais
            Af = Regression.sigmoid(output)

            dl_dw += np.matmul(Af - Y[i,:], self.input[:,i].T)
            dl_db += np.matmul(Af- Y[i,:],np.ones(self.bais.shape).T)
            dl_back[:,i] = (np.matmul(Af - Y[i,:], self.weight)).T

        self.weight = self.weight - (learning_rate*dl_dw/Y.shape[1])
        self.bais = self.bais - (learning_rate*dl_db/Y.shape[1])

        return dl_back

    