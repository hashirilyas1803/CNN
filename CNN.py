import os
from PIL import Image
import numpy as np
from Regression import Regression
from Convolution_Layer import Convolution_Layer
from Pooling import Pooling
from Fully_Connected import fully_connected
from intermediate_connect import intermediate_connect

def main():
    if os.path.exists('arrays.npz'):
        # If the file exists, load the data
        data = np.load('arrays.npz')
        X = data['X']
        y = data['y']
    else:
        X, y = load_data("sign_data/train")
    X = np.expand_dims(X, axis=1)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    alpha = 0.01
    c_layer, p_layer, f_layer = initialize_layers()
    #Train
    # train_model(X,y,c_layer, p_layer, f_layer,alpha,10)
    #Test
    print(test_model())

def load_data(path):
    # Create a list to hold all your images
    data = []

    # Get a sorted list of all subdirectories
    subdirs = sorted([os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    # Iterate over every alternate subdirectory
    for i in range(0, len(subdirs), 2):
        # Load genuine signatures
        for filename in os.listdir(subdirs[i]):
            with Image.open(os.path.join(subdirs[i], filename)) as img:
                img = img.convert('L')
                img = img.resize((64, 64))
                img_data = np.array(img)
                data.append((img_data, 0))

        # Check if there is a next directory and load forged signatures
        if i+1 < len(subdirs):
            for filename in os.listdir(subdirs[i+1]):
                with Image.open(os.path.join(subdirs[i+1], filename)) as img:
                    img = img.convert('L')
                    img = img.resize((64, 64))
                    img_data = np.array(img)
                    data.append((img_data, 1))
    
    images, y = zip(*data)
    y = np.array(y).reshape(-1, 1)
    # Convert list to numpy array
    X = np.array(images)
    np.savez('arrays.npz', X=X, y=y)
    return X, y

def initialize_layers():
    c_layer = Convolution_Layer(1,3)
    p_layer = Pooling(2)
    f_layer = fully_connected(1)
    return c_layer,p_layer,f_layer

def flatten_output(A):
    # Flatten the output
    m = A.shape[0]
    flattened = A.reshape(m, -1)
    return flattened.T

def train_model(X, y, c_layer, p_layer, f_layer, alpha, epochs):
    for epoch in range(epochs):
        # Forward Propagation
        A = c_layer.forward_prop(X)
        A = Regression.activate(A)
        Z = p_layer.forward_prop(A)
        Z = flatten_output(Z)
        A = f_layer.forward_prop(Z)
        A = Regression.sigmoid(A)

        #Backward Propagation
        dl_back = f_layer.backward_prop(alpha, y)
        dl_back = dl_back.T.reshape(-1, 1, A.shape[1])
        dl_back = p_layer.backward_prop(dl_back)
        dl_back = c_layer.backward_prop(dl_back, alpha)

def load_test_data(path):
    data = []
    subdirs = sorted([os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    for i in range(0, len(subdirs), 2):
        for filename in os.listdir(subdirs[i]):
            with Image.open(os.path.join(subdirs[i], filename)) as img:
                img = img.convert('L')
                img = img.resize((64, 64))
                img_data = np.array(img)
                data.append((img_data, 0))

        if i+1 < len(subdirs):
            for filename in os.listdir(subdirs[i+1]):
                with Image.open(os.path.join(subdirs[i+1], filename)) as img:
                    img = img.convert('L')
                    img = img.resize((64, 64))
                    img_data = np.array(img)
                    data.append((img_data, 1))
    
    images, y = zip(*data)
    y = np.array(y).reshape(-1, 1)
    X = np.array(images)
    X = np.expand_dims(X, axis=1)
    return X, y

def test_one(path):
    c_layer, p_layer,f_layer = initialize_layers
    X_test, y_test = load_test_data(path)
    A = c_layer.forward_prop(X_test)
    A = Regression.activate(A)
    Z = p_layer.forward_prop(A)
    Z = flatten_output(Z)
    A = f_layer.forward_prop(Z)
    A = Regression.sigmoid(A)

    # Output
    prediction = (A > 0.5).astype(int)
    return prediction
    

def test_model():
    path = "sign_data/test"
    c_layer, p_layer,f_layer = initialize_layers()
    X_test, y_test = load_test_data(path)
    A = c_layer.forward_prop(X_test)
    A = Regression.activate(A)
    Z = p_layer.forward_prop(A)
    Z = flatten_output(Z)
    A = f_layer.forward_prop(Z)
    A = Regression.sigmoid(A)

    # Calculate accuracy
    predictions = (A > 0.4).astype(int)
    accuracy = np.mean(predictions == y_test)
    return f"{accuracy * 100:.2f}"

if __name__ == "__main__":
    main()