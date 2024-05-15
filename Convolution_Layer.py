import numpy as np
import os
from numpy.lib.stride_tricks import as_strided

class Convolution_Layer:
    layer_count = 0  # Class-level variable to keep track of the number of instances

    def __init__(self, num_filters, filter_size):
        Convolution_Layer.layer_count += 1
        self.layer_id = f'Convolution_Layer_{Convolution_Layer.layer_count}'  # Unique identifier for each instance
        self.num_filters = num_filters
        self.filter_size = filter_size

        # Initialize filters and biases
        self.conv_filter = np.random.randn(num_filters,filter_size,filter_size) / (filter_size ** 2)
        self.biases = np.zeros(num_filters)
        # self.conv_filter = [[1, 0, -1],
        #                     [1, 0, -1],
        #                     [1, 0, -1]]
        # self.conv_filter = np.array(self.conv_filter)

    # def forward_prop(self, A_prev):
    #     # Get dimensions
    #     (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    #     f, n_C, k, k = self.conv_filter.shape

    #     # Ensure consistency with input channels
    #     if n_C_prev != n_C:
    #         raise ValueError("Number of input channels (n_C_prev) does not match filter channel depth (n_C)")

    #     # Calculate output height and width (considering stride of 1)
    #     n_H = n_H_prev - k + 1
    #     n_W = n_W_prev - k + 1

    #     # Initialize output matrix
    #     Z = np.zeros((m, f, n_H, n_W))

    #     # Perform convolution for each filter
    #     for i in range(f):
    #         for h in range(n_H):
    #             for w in range(n_W):
    #                 # Extract a slice of the input for the current filter
    #                 a_slice = A_prev[:, :, h:h+k, w:w+k]

    #                 # Element-wise multiplication between filter and input slice
    #                 Z[m, i, h, w] = np.sum(a_slice * self.conv_filter[i]) + self.biases[i]

    #     return Z

    def forward_prop(self, X):
        self.input = X
        # Dimensions of input X
        m, channels, iH, iW = X.shape

        # Dimensions of the filter
        kH, kW = self.filter_size, self.filter_size

        # Size of output
        oH = iH - kH + 1
        oW = iW - kW + 1

        # Ensure window shape does not exceed input shape
        if kH > iH or kW > iW:
            raise ValueError(f"Window shape ({kH}, {kW}) cannot be larger than input shape ({iH}, {iW}).")

        # Create image patches
        patches_shape = (m, channels, oH, oW, kH, kW)
        patches_strides = (X.strides[0], X.strides[1], X.strides[2], X.strides[3], X.strides[2], X.strides[3])
        patches = as_strided(X, shape=patches_shape, strides=patches_strides)

        # Reshape patches to align with dot operation
        patches = patches.reshape(m * oH * oW, channels * kH * kW)

        # Reshape filters for the dot operation
        filters_reshaped = self.conv_filter.reshape(self.num_filters, channels * kH * kW).T

        # Convolution operation: matrix multiplication plus bias
        convolved = np.dot(patches, filters_reshaped).reshape(m, oH, oW, self.num_filters)

        # Adding the bias to each filter result
        convolved += self.biases[np.newaxis, np.newaxis, np.newaxis, :]

        # Transpose the result to match the output shape (m, num_filters, oH, oW)
        output = convolved.transpose(0, 3, 1, 2)

        return output
        # self.input = X
        # # Dimensions of input X
        # m, channels, iH, iW = X.shape

        # if self.conv_filter is None:
        #     # Initialize filters considering the number of input channels
        #     self.conv_filter = np.random.randn(self.num_filters, channels, self.filter_size, self.filter_size) / (self.filter_size * self.filter_size)
        #     # Initialize biases, one for each filter
        #     self.biases = np.zeros(self.num_filters)

        # # Dimensions of the filter
        # kH, kW = self.filter_size, self.filter_size
        
        # # Size of output
        # oH = iH - kH + 1
        # oW = iW - kW + 1

        # # Ensure window shape does not exceed input shape
        # if kH > iH or kW > iW:
        #     raise ValueError(f"Window shape ({kH}, {kW}) cannot be larger than input shape ({iH}, {iW}).")
        
        # window_shape = (1, channels, kH, kW)

        # # Create image patches
        # patches = np.lib.stride_tricks.sliding_window_view(X, window_shape)
        # patches = patches.reshape(m, oH, oW, channels * kH * kW)

        # # Reshape patches to align with dot operation
        # patches = patches.reshape(m * oH * oW, -1)

        # # Reshape filters for the dot operation
        # filters_reshaped = self.conv_filter.reshape(self.num_filters, -1).T

        # # Convolution operation: matrix multiplication plus bias
        # convolved = np.dot(patches, filters_reshaped).reshape(m, oH, oW, self.num_filters)

        # # Adding the bias to each filter result
        # convolved += self.biases[np.newaxis, np.newaxis, np.newaxis, :]

        # # Transpose the result to match the output shape (m, num_filters, oH, oW)
        # return convolved.transpose(0, 3, 1, 2)
    
    def backward_prop(self, dL_dA, learning_rate):
        m, channels, iH, iW = self.input.shape
        kH, kW = self.filter_size, self.filter_size
        _, num_filters, oH, oW = dL_dA.shape

        # Initialize gradients
        dL_dW = np.zeros_like(self.conv_filter)
        dL_db = np.zeros_like(self.biases)
        dL_dX = np.zeros_like(self.input)

        # Create image patches
        patches_shape = (m, oH, oW, channels, kH, kW)
        patches_strides = (self.input.strides[0], self.input.strides[2], self.input.strides[3], self.input.strides[1], self.input.strides[2], self.input.strides[3])
        patches = as_strided(self.input, shape=patches_shape, strides=patches_strides)
        patches = patches.reshape(m * oH * oW, channels * kH * kW)

        # Reshape filters for the dot operation
        filters_reshaped = self.conv_filter.reshape(self.num_filters, channels * kH * kW)

        # Gradient of the loss w.r.t. filters
        dL_dA_reshaped = dL_dA.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        dL_dW = np.dot(dL_dA_reshaped, patches).reshape(self.conv_filter.shape)

        # Gradient of the loss w.r.t. biases
        dL_db = np.sum(dL_dA, axis=(0, 2, 3))

        # Gradient of the loss w.r.t. the input
        filters_reshaped_T = filters_reshaped.T
        dL_dA_reshaped = dL_dA.transpose(1, 0, 2, 3).reshape(num_filters, -1)
        for i in range(oH):
            for j in range(oW):
                h_start = i
                w_start = j
                h_end = h_start + kH
                w_end = w_start + kW

                patch = dL_dA[:, :, i, j].reshape(m, num_filters, 1, 1)
                dL_dX[:, :, h_start:h_end, w_start:w_end] += np.sum(patch * filters_reshaped_T.T[:, :, None, None], axis=1)

        # Update weights and biases
        self.conv_filter -= learning_rate * dL_dW
        self.biases -= learning_rate * dL_db

        # Save updated weights
        self.save_weights()

        return dL_dX

    def save_weights(self):
        weights_path = 'weights.npz'
        if os.path.exists(weights_path):
            data = np.load(weights_path)
            weight_dict = {k: data[k] for k in data}
            weight_dict[f'weight_{self.layer_id}'] = self.conv_filter
            weight_dict[f'bias_{self.layer_id}'] = self.biases
            np.savez(weights_path, **weight_dict)
        else:
            np.savez(weights_path, **{f'weight_{self.layer_id}': self.conv_filter, f'bias_{self.layer_id}': self.biases})