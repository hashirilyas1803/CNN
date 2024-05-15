import os
from PIL import Image
import numpy as np
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
    print(X.shape)
    
    # Initialize model components
    conv_layers, pool_layers, intermediate_connect_layer, fully = initialize_layers()
    
    # Train the model
    train_model(X, y, conv_layers, pool_layers, intermediate_connect_layer, fully, alpha=0.1, epochs=10)
    
    # Test the model
    test_model("sign_data/test", conv_layers, pool_layers, intermediate_connect_layer, fully)

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
    conv_layers = []
    pool_layers = []
    l = []
    x = 30
    for i in range(15):
        l.append(x)
        x -= 2
    stride = [1, 2]

    conv_layers.append(Convolution_Layer(l[0], 3))
    for i in range(1, 15):
        conv_layers.append(Convolution_Layer(l[i], 3))
        if i < 8:
            pool_layers.append(Pooling(3, stride[1]))
        else:
            pool_layers.append(Pooling(3, stride[0]))

    n = [10, 8, 4, 2]
    intermediate_connect_layer = []
    fully = fully_connected(1)
    for i in range(4):
        intermediate_connect_layer.append(intermediate_connect(n[i]))

    return conv_layers, pool_layers, intermediate_connect_layer, fully

def flatten_output(A):
    # Flatten the output
    m = A.shape[0]
    flattened = A.reshape(m, -1)
    return flattened

def train_model(X, y, conv_layers, pool_layers, intermediate_connect_layer, fully, alpha=0.1, epochs=10):
    for epoch in range(epochs):
        # Forward Propagation
        Z = conv_layers[0].forward_prop(X)
        A = pool_layers[0].forward_prop(Z)
        for j in range(1, 15):
            Z = conv_layers[j].forward_prop(A)
            A = pool_layers[j].forward_prop(Z)

        # Flatten the output
        A = flatten_output(A)

        for j in range(4):
            A = intermediate_connect_layer[j].forward_prop(A)

        A = fully.forward_prop(A)

        # Backward Propagation
        dl_back = fully.backward_prop(alpha, y)

        for i in range(3, -1, -1):
            dl_back = intermediate_connect_layer[i].backward_prop(alpha, dl_back)

        dl_back = dl_back.T.reshape(-1, 1, A.shape[1])
        
        for j in range(14, -1, -1):
            dl_back = pool_layers[j].backward_prop(dl_back)
            dl_back = conv_layers[j].backward_prop(dl_back)

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

def test_model(path, conv_layers, pool_layers, intermediate_connect_layer, fully):
    X_test, y_test = load_test_data(path)
    # Forward Propagation
    Z = conv_layers[0].forward_prop(X_test)
    A = pool_layers[0].forward_prop(Z)
    for j in range(1, 15):
        Z = conv_layers[j].forward_prop(A)
        A = pool_layers[j].forward_prop(Z)

    # Flatten the output
    A = flatten_output(A)

    for j in range(4):
        A = intermediate_connect_layer[j].forward_prop(A)

    A = fully.forward_prop(A)

    # Calculate accuracy
    predictions = (A > 0.5).astype(int)
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()