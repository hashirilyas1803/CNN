import numpy as np
import CNNS.Regression as Regression
import math

class Convolution_Operation:

    def _init_(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filer_size = filter_size
        self.conv_filter = np.random.randn(num_filters,filter_size,filter_size)/(filter_size)**2

