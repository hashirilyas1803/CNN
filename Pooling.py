import numpy as np
from numpy.lib.stride_tricks import as_strided

class Pooling:

    def __init__(self,filter_size,stride):
        self.filter_size = filter_size
        self.stride = stride
    
    def forward_prop(self,X):
        self.input = X
        m, channels, iH, iW = X.shape
        oH = (iH - self.pool_size) // self.stride + 1
        oW = (iW - self.pool_size) // self.stride + 1

        # Create a sliding window view of the input
        shape = (m, channels, oH, oW, self.pool_size, self.pool_size)
        strides = (X.strides[0], X.strides[1], X.strides[2]*self.stride, X.strides[3]*self.stride, X.strides[2], X.strides[3])
        X_strided = as_strided(X, shape=shape, strides=strides)

        # Perform max pooling in a vectorized manner
        output = X_strided.max(axis=(4, 5))

        return output

    def backward_prop(self,dl_back):

        if len(dl_back.shape) != len(self.input.shape):
            p = dl_back.reshape(self.input.shape[0],self.input.shape[1],self.filter_size,self.filter_size)

        dl_back = np.zeros(self.input.shape)

        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                for k in range(self.input.shape[3]/self.filter_size):
                    for l in range(self.input.shape[3]/self.filter_size):
                        block = self.input[i, j, k * self.input.shape[3] / self.filter_size:(k+1) * self.input.shape[3] / self.filter_size, l * self.input.shape[3] / self.filter_size:(l+1) * self.input.shape[3] / self.filter_size]
                        max_index_flat = np.argmax(block)
                        max_index_2d = np.unravel_index(max_index_flat, block.shape)
                        dl_back[i,j,max_index_2d[0],max_index_2d[1]] = p[i,j,max_index_2d[0],max_index_2d[1]]

        return dl_back

    