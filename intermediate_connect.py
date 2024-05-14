import numpy as np
from Regression import Regression

class intermediate_connect:

    def __init__(self,input_size,softmax_size):
        self.input = input
        self.weight = np.random.randn(softmax_size,input_size)/input_size
        self.bais = np.zeros(softmax_size,1)
        
    def backward_prop(self,learning_rate,dl_back):
    
        dA = dl_back

        dl_dw = np.zeros(self.weight.shape) 
        dl_db = np.zeros(self.bais.shape)
        dl_back = np.zeros(self.input.shape)
        
        for i in range(self.input.shape[2]):

            output = np.matmul(self.weight,self.input[:,:,i]) + self.bais
            dZ = Regression.der_sigmoid(output)*dA[:,:,i]

            dl_dw += np.matmul(dZ, self.input[:,:,i].T)
            dl_db += np.matmul(dZ,np.ones(self.bais.shape).T)
            dl_back[:,:,i] = (np.matmul(dZ, self.weight)).T

        self.weight = self.weight - (learning_rate*dl_dw/self.input.shape[2])
        self.bais = self.bais - (learning_rate*dl_db/self.input.shape[2])

        return dl_back