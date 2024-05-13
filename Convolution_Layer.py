import numpy as np

class Convolution_Layer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters, filter_size, filter_size) / filter_size**2
    
    def forward_prop(self, X):
        # Dimensions of input X
        iH, iW = X.shape

        # Dimensions of the filter
        kH, kW = self.filter_size, self.filter_size

        # Size of output
        oH = iH - kH + 1
        oW = iW - kW + 1

        # Create an empty output for each filter
        output = np.zeros((self.num_filters, oH, oW))

        # Loop through each filter
        for idx in range(self.num_filters):
            # Extract the current filter
            current_filter = self.conv_filter[idx, :, :]

            # Create X patches and apply the current filter
            patches = np.lib.stride_tricks.sliding_window_view(X, (kH, kW))
            output[idx] = np.einsum('ijkl,kl->ij', patches, current_filter)

        return output