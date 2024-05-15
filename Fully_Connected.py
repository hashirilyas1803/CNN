import numpy as np
import os
from Regression import Regression

class fully_connected:
    layer_count = 0  # Class-level variable to keep track of the number of instances

    def __init__(self, number_of_neurons):
        fully_connected.layer_count += 1
        self.layer_id = f'fully_connected_{fully_connected.layer_count}'  # Unique identifier for each instance
        self.number_of_neurons = number_of_neurons

    def forward_prop(self, X):
        self.input = X
        weights_path = 'weights.npz'
        
        if os.path.exists(weights_path):
            data = np.load(weights_path)
            self.weight = data.get(f'weight_{self.layer_id}')
            self.bias = data.get(f'bias_{self.layer_id}')
        else:
            self.weight = np.random.randn(self.number_of_neurons, X.shape[0]) / X.shape[0] - 0.5
            self.bias = np.zeros((self.number_of_neurons, 1)) - 0.5
        
        # Perform the matrix multiplication
        Z = np.matmul(self.weight, X) + self.bias

        return Z
    
    def backward_prop(self, learning_rate, Y):
        dl_dw = np.zeros(self.weight.shape) 
        dl_db = np.zeros(self.bias.shape)
        dl_back = np.zeros(self.input.shape)

        for i in range(self.input.shape[1]):
            output = np.matmul(self.weight, self.input[:, i]) + self.bias
            Af = Regression.sigmoid(output)

            error = Af - Y[i, :].reshape(-1, 1)

            dl_dw += np.matmul(error, self.input[:, i].reshape(1, -1))
            dl_db += error
            dl_back[:, i] = np.matmul(self.weight.T, error).flatten()

        self.weight -= learning_rate * dl_dw / Y.shape[0]
        self.bias -= learning_rate * dl_db / Y.shape[0]
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
