import os
from PIL import Image
import numpy as np
import Convolution_Layer
import Pooling
import Fully_Connected

def main():
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

    conv_layers.append(Convolution_Layer(l[i],3,1))
    for i in range(15):
        conv_layers.append(Convolution_Layer(l[i],3,conv_layers[i].channels))
        if i < 8:
            pool_layers.append(Pooling(3,stride[1]))
        else:
            pool_layers.append(Pooling(3,stride[0]))
    # TODO
    fully_connected = []
    for i in range(5):
        pass



        


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