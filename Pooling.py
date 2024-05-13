import numpy as np

class Pooling:

    def __init__(self,filter_size,input):
        self.filter_size = filter_size
        self.input = input

    def backward_prop(self,dl_out):

        p = dl_out.reshape(self.input.shape[0],self.input.shape[1],self.input.shape[2],self.input.shape[3])
        print(p)

        return

    