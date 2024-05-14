import os
from PIL import Image
import numpy as np
import Convolution_Layer
import Pooling
import Fully_Connected
import intermediate_connect

def main():
    # Learning Rate
    alpha = 0.1
    if os.path.exists('arrays.npz'):
        # If the file exists, load the data
        data = np.load('arrays.npz')
        X = data['X']
        y = data['y']
    else:
        X,y = load_data("sign_data/train")
    X = np.expand_dims(X, axis=1)

    conv_layers = []
    pool_layers = []
    l = []
    x = 30
    for i in range(15):
        l.append(x)
        x -= 2
    stride = [1,2]

    conv_layers.append(Convolution_Layer(l[0],3,1))
    for i in range(1,15):
        conv_layers.append(Convolution_Layer(l[i],3,conv_layers[i].channels))
        if i < 8:
            pool_layers.append(Pooling(3,stride[1]))
        else:
            pool_layers.append(Pooling(3,stride[0]))
    # TODO
    fully_connected = []
    for i in range(5):
        pass

    n = [10,8,4,2]
    for i in range(10):
        # Forward Propagation
        Z = conv_layers[0].forward_prop(X)
        A = pool_layers[0].forward_prop(Z)
        for j in range(1,15):
            Z = conv_layers[j].forward_prop(A)
            A = pool_layers[j].forward_prop(Z)
        for j in range(4):
            if j != 4:
                A = intermediate_connect[j].forward_prop(A,j)
            if(j == 4):
                A = fully.forward_prop(A,1)

        # Backward Propagation
        dl_back = fully.backward_prop(alpha,y)

        for i in range(4,0,-1):
            dl_back = intermediate_connect[i].backward_prop(alpha,dl_back)
        for j in range(15,1,-1):
            dl_back = pool_layers[j].backward_prop(dl_back)
            dl_back = conv_layers[j].backward_prop(dl_back)
            
        dl_back = pool_layers[0].backward_prop(dl_back)
        dl_back = conv_layers[0].bacward_prop(dl_back)
    
def load_data(path):
    # create a list to hold all your images
    data = []

    # get a sorted list of all subdirectories
    subdirs = sorted([os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    # iterate over every alternate subdirectory
    for i in range(0, len(subdirs), 2):
        # load genuine signatures
        for filename in os.listdir(subdirs[i]):
            with Image.open(os.path.join(subdirs[i], filename)) as img:
                img = img.convert('L')
                img = img.resize((64, 64))
                img_data = np.array(img)
                data.append((img_data, 0)) 

        # check if there is a next directory and load forged signatures
        if i+1 < len(subdirs):
            for filename in os.listdir(subdirs[i+1]):
                with Image.open(os.path.join(subdirs[i+1], filename)) as img:
                    img = img.convert('L')
                    img = img.resize((64, 64))
                    img_data = np.array(img)
                    data.append((img_data, 1))
    
    images, y = zip(*data)
    y = np.array(y).reshape(-1,1)
    # convert list to numpy array
    X = np.array(images)
    np.savez('arrays.npz', X=X, y=y)
    return X,y

if __name__ == "__main__":
    main()