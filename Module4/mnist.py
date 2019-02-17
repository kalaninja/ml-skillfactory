
import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def split_labels_features(dataset):
    labels = []
    features = []
    for label_features in dataset:
        labels.append(label_features[0])
        features.append(np.ndarray.flatten(label_features[1]))
    return labels, features

def read_all(path):
    labels_features_train = list(read('training', path))
    labels_train, data_train = split_labels_features(labels_features_train) 
    labels_features_test = list(read('testing', path))
    labels_test, data_test = split_labels_features(labels_features_test) 
    return data_train, labels_train, data_test, labels_test

def read_filtered_labels(labels, path):
    labels_features_train = list(filter(lambda row: row[0] in labels, read('training', path)))
    labels_features_test = list(filter(lambda row: row[0] in labels, read('testing', path)))
    labels_train, data_train = split_labels_features(labels_features_train) 
    labels_test, data_test = split_labels_features(labels_features_test) 
    return data_train, labels_train, data_test, labels_test

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image.reshape(28, 28), cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()
