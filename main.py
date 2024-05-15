import numpy as np
from Fully_Connected import fully_connected
from Convolution_Layer import Convolution_Layer

def create_random_array(shape, num_ones):
    # Create an array of zeros with the specified shape
    array = np.zeros(shape, dtype=int)
    
    # Choose random indices for the ones, ensuring no duplicates
    indices = np.random.choice(np.prod(shape), size=num_ones, replace=False)
    
    # Set the chosen positions to one
    np.put(array, indices, 1)
    
    return array

print(len((2,5,5,5)))

array = [[3, 0, 1, 2, 7, 4],
         [1, 5, 8, 9, 3, 1],
         [2, 7, 2, 5, 1, 3],
         [0, 1, 3, 1, 7, 8],
         [4, 2, 1, 6, 2, 8],
         [2, 4, 5, 2, 3, 9]]
array = np.array(array)
array = np.expand_dims(array, axis=0)
array = np.expand_dims(array, axis=0)
print(array.shape)
layer = Convolution_Layer(1, 3)
print(layer.forward_prop(array).shape)

# ran = np.random.randn(10,1,30)/10

# test = create_random_array((30,1) , 10)

# fully = fully_connected(10, 1, ran)

# result = fully.backward_prop(0.1,test)

# print(result)


