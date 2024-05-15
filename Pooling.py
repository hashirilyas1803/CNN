import numpy as np
from numpy.lib.stride_tricks import as_strided

class Pooling:

    def __init__(self,stride):
        self.stride = stride
    
    def forward_prop(self, X):
        self.input = X
        self.pool_size = int(X.shape[2] / 2)
        m, channels, iH, iW = X.shape
        kH, kW = self.pool_size, self.pool_size

        # Calculate the dimensions of the output
        oH = (iH - kH) // self.stride + 1
        oW = (iW - kW) // self.stride + 1

        # Create an output array filled with negative infinity (since we are doing max pooling)
        output = np.full((m, channels, oH, oW), -np.inf)

        for i in range(oH):
            for j in range(oW):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = min(h_start + kH, iH)
                w_end = min(w_start + kW, iW)
                patch = X[:, :, h_start:h_end, w_start:w_end]
                output[:, :, i, j] = np.max(patch, axis=(2, 3))

        # Handle remaining edges if they exist
        if (iH - kH) % self.stride != 0:
            h_start = oH * self.stride
            if h_start < iH:
                for j in range(oW):
                    w_start = j * self.stride
                    w_end = min(w_start + kW, iW)
                    patch = X[:, :, h_start:iH, w_start:w_end]
                    output[:, :, oH-1, j] = np.maximum(output[:, :, oH-1, j], np.max(patch, axis=(2, 3)))

        if (iW - kW) % self.stride != 0:
            w_start = oW * self.stride
            if w_start < iW:
                for i in range(oH):
                    h_start = i * self.stride
                    h_end = min(h_start + kH, iH)
                    patch = X[:, :, h_start:h_end, w_start:iW]
                    output[:, :, i, oW-1] = np.maximum(output[:, :, i, oW-1], np.max(patch, axis=(2, 3)))

        self.output = output
        return output

    def backward_prop(self,dl_back):

        if len(dl_back.shape) != len(self.input.shape):
            p = dl_back.reshape(self.output.shape)

        dl_back = np.zeros(self.input.shape)

        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                for k in range(int(self.input.shape[3]/self.pool_size)):
                    for l in range(int(self.input.shape[3]/self.pool_size)):
                        block = self.input[i, j, k * int(self.input.shape[3] / self.pool_size):(k+1) * int(self.input.shape[3] / self.pool_size), l * int(self.input.shape[3] / self.pool_size):(l+1) * int(self.input.shape[3] / self.pool_size)]
                        max_index_flat = np.argmax(block)
                        max_index_2d = np.unravel_index(max_index_flat, block.shape)
                        dl_back[i,j,max_index_2d[0],max_index_2d[1]] = p[i,j,max_index_2d[0],max_index_2d[1]]

        return dl_back

    