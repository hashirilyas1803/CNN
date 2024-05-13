import numpy as np
from numpy.lib.stride_tricks import as_strided

class Pooling:

    def __init__(self,filter_size,input,stride):
        self.filter_size = filter_size
        self.input = input
        self.stride = stride
    
    def forward_prop(self,X):
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

    def backward_prop(self,dl_out):

        p = dl_out.reshape(self.input.shape[0],self.input.shape[1],self.input.shape[2],self.input.shape[3])
        print(p)

        return

    