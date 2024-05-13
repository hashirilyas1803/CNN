import numpy as np
import math 
from Regression import Regression

class fully_connected:

    def __init__(self,input_size,softmax_size,input):
        self.input = input
        self.weight = np.random.randn(softmax_size,input_size)/input_size
        self.bias = np.zeros(softmax_size)

    def forward_prop(self, X):
        Z = np.dot(self.weight, X) + self.bias
        A = Regression.relu(Z)
        return A
    
    def backward_prop(self,learning_rate,Y):
    
        dl_dw = np.zeros(self.weight.shape) 
        dl_db = np.zeros(self.bias.shape)
        dl_back = np.zeros(self.input.shape)
        print(dl_back.shape)
        for i in range(self.input.shape[2]):

            output = np.matmul(self.weight,self.input[:,:,i]) + self.bias
            Af = Regression.softmax(self,output)

            dl_dw += np.matmul(Af - Y[i,:], self.input[:,:,i].T)
            dl_db += np.matmul(Af- Y[i,:],np.ones(self.bias.shape).T)
            dl_back[:,:,i] = (np.matmul(Af - Y[i,:], self.weight)).T

        self.weight = self.weight - (learning_rate*dl_dw/Y.shape[1])
        self.bias = self.bias - (learning_rate*dl_db/Y.shape[1])
        print(self.weight)
        return dl_back

    