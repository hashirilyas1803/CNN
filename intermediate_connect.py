import numpy as np
import os
from Regression import Regression

class intermediate_connect:
    layer_count = 0  # Class-level variable to keep track of the number of instances

    def __init__(self, number_of_neurons):
        intermediate_connect.layer_count += 1
        self.layer_id = f'intermediate_connect_{intermediate_connect.layer_count}'  # Unique identifier for each instance
        self.number_of_neurons = number_of_neurons

    def forward_prop(self, X):
        self.input = X
        weights_path = 'weights.npz'
        
        if os.path.exists(weights_path):
            data = np.load(weights_path)
            self.weight = data.get(f'weight_{self.layer_id}', np.random.randn(self.number_of_neurons, X.shape[0]) / X.shape[0])
            self.bias = data.get(f'bias_{self.layer_id}', np.zeros((self.number_of_neurons, 1)))
        else:
            self.weight = np.random.randn(self.number_of_neurons, X.shape[0]) / X.shape[0]
            self.bias = np.zeros((self.number_of_neurons, 1))
        
        Z = np.matmul(self.weight, X) + self.bias
        A = np.maximum(0, Z)  # ReLU activation function
        return A
        
    def backward_prop(self, learning_rate, dl_back):
        dA = dl_back
        dl_dw = np.zeros(self.weight.shape) 
        dl_db = np.zeros(self.bias.shape)
        dl_back = np.zeros(self.input.shape)
        
        for i in range(self.input.shape[1]):
            output = np.matmul(self.weight, self.input[:, i]) + self.bias
            dZ = Regression.der_sigmoid(output) * dA[:, i]
            dl_dw += np.matmul(dZ, self.input[:, i].T)
            dl_db += np.matmul(dZ, np.ones(self.bias.shape).T)
            dl_back[:, i] = (np.matmul(dZ, self.weight)).T

        self.weight -= learning_rate * dl_dw / self.input.shape[1]
        self.bias -= learning_rate * dl_db / self.input.shape[1]
        self.save_weights()

        return dl_back

    def save_weights(self):
        weights_path = 'weights.npz'
        if os.path.exists(weights_path):
            data = np.load(weights_path)
            weight_dict = {k: data[k] for k in data}
            weight_dict[f'weight_{self.layer_id}'] = self.weight
            weight_dict[f'bias_{self.layer_id}'] = self.bias
            np.savez(weights_path, **weight_dict)
        else:
            np.savez(weights_path, **{f'weight_{self.layer_id}': self.weight, f'bias_{self.layer_id}': self.bias})