import numpy as np
import math 
from Regression import Regression

class fully_connected:

   def __init__(self, input_size, softmax_size):
    self.weight = np.random.randn(softmax_size, 1, input_size) / input_size
    self.bias = np.zeros((softmax_size, 1, 1))

    def forward_prop(self, X):
        # Perform the matrix multiplication
        Z = np.matmul(self.weight, X) + self.bias

        # Apply the ReLU activation function
        A = np.maximum(0, Z)

        return A
    
    def backward_prop(self,learning_rate,Y):
        dl_dw = np.zeros(self.weight.shape) 
        dl_db = np.zeros(self.bias.shape)
        dl_back = np.zeros(self.input.shape)
        print(dl_back.shape)
        for i in range(self.input.shape[2]):
            output = np.matmul(self.weight,self.input[:,:,i]) + self.bias
            Af = Regression.sigmoid(self,output)

            dl_dw += np.matmul(Af - Y[i,:], self.input[:,:,i].T)
            dl_db += np.matmul(Af- Y[i,:],np.ones(self.bias.shape).T)
            dl_back[:,:,i] = (np.matmul(Af - Y[i,:], self.weight)).T

        self.weight = self.weight - (learning_rate*dl_dw/Y.shape[1])
        self.bias = self.bias - (learning_rate*dl_db/Y.shape[1])
        print(self.weight)
        return dl_back

    