import numpy as np
import gzip
import struct

def load_images(filename):

    with gzip.open(filename, 'rb') as f:
       
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all the pixels into a NumPy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)

def prepend_bias(X):
    return np.insert(X, 0, 1, axis=1)

X_train = prepend_bias(load_images("mnist\\train-images-idx3-ubyte.gz"))
#X_train = prepend_bias(load_images("D:\\Data\\Pycode\\Paollo\\mnist\\train-images-idx3-ubyte.gz"))

X_test = prepend_bias(load_images("mnist\\t10k-images-idx3-ubyte.gz"))

def load_labels(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all the labels into a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix:
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def encode_fives(Y):
    # Convert all 5s to 1, and everything else to 0
    return (Y == 5).astype(int)


# 60K labels, each with value 1 if the digit is a five, and 0 otherwise
Y_train = encode_fives(load_labels("mnist\\train-labels-idx1-ubyte.gz"))

# 10000 labels, with the same encoding as Y_train
Y_test = encode_fives(load_labels("mnist\\t10k-labels-idx1-ubyte.gz"))
