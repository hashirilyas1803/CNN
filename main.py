import numpy as np
from Fully_Connected import fully_connected

def create_random_array(shape, num_ones):
    # Create an array of zeros with the specified shape
    array = np.zeros(shape, dtype=int)
    
    # Choose random indices for the ones, ensuring no duplicates
    indices = np.random.choice(np.prod(shape), size=num_ones, replace=False)
    
    # Set the chosen positions to one
    np.put(array, indices, 1)
    
    return array


ran = np.random.randn(10,1,30)/10

test = create_random_array((30,1) , 10)

fully = fully_connected(10, 1, ran)

result = fully.backward_prop(0.1,test)

print(result)


