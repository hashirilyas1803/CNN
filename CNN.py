import os
from PIL import Image
import numpy as np

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
                data.append(img_data)

        # check if there is a next directory and load forged signatures
        if i+1 < len(subdirs):
            for filename in os.listdir(subdirs[i+1]):
                with Image.open(os.path.join(subdirs[i+1], filename)) as img:
                    img = img.convert('L')
                    img = img.resize((64, 64))
                    img_data = np.array(img)
                    data.append(img_data)

    # convert list to numpy array
    X = np.array(data)
    return X

X = load_data("sign_data/train")
X = np.expand_dims(X, axis=1)