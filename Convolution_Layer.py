import numpy as np

class Convolution_Layer:
    def __init__(self, num_filters, filter_size, num_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.num_channels = num_channels  # This now needs to be explicitly provided during initialization
        # Initialize filters considering the number of input channels
        self.conv_filter = np.random.randn(num_filters, num_channels, filter_size, filter_size) / (filter_size * filter_size)
        # Initialize biases, one for each filter
        self.biases = np.zeros(num_filters)

    def forward_prop(self, X):
        # Dimensions of input X
        m, channels, iH, iW = X.shape
        
        # Check if the input channels match the expected number of channels
        assert channels == self.num_channels, "Input channel mismatch."
        
        # Dimensions of the filter
        kH, kW = self.filter_size, self.filter_size
        
        # Size of output
        oH = iH - kH + 1
        oW = iW - kW + 1
        
        # Create image patches
        patches = np.lib.stride_tricks.sliding_window_view(X, window_shape=(channels, kH, kW))
        patches = patches.reshape(m, oH, oW, channels * kH * kW)
        
        # Reshape patches to align with dot operation
        patches = patches.reshape(m * oH * oW, kH * kW * channels)

        # Reshape filters for the dot operation
        filters_reshaped = self.conv_filter.reshape(self.num_filters, kH * kW * channels).T

        # Convolution operation: matrix multiplication plus bias
        convolved = np.dot(patches, filters_reshaped).reshape(m, oH, oW, self.num_filters)
        
        # Adding the bias to each filter result
        convolved += self.biases[np.newaxis, np.newaxis, np.newaxis, :]

        # Transpose the result to match the output shape (m, num_filters, oH, oW)
        return convolved.transpose(0, 3, 1, 2)