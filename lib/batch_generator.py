import numpy as np

def batch_generator(X, y, batch_size, shuffle=True):
    """
    Batch generator, X is a list contains inputs from different sources
    y is a numpy array
    """
    size = X[0].shape[0]
    indices = np.arange(size)
    if shuffle == True:
        np.random.shuffle(indices)
    X_copy = []
    for i in range(len(X)):
        X_copy.append(X[i][indices])
    y_copy = y[indices]
    i = 0
    while True:
        if i + batch_size < size:
            x_batch = []
            y_batch = y_copy[i:i + batch_size]
            for j in range(len(X)):
                x_batch.append(X_copy[j][i:i + batch_size])
            yield x_batch, y_batch
            i += batch_size
        else:
            x_batch = []
            y_batch = y_copy[i:size]
            for j in range(len(X)):
                x_batch.append(X_copy[j][i:size])
            yield x_batch, y_batch
            i = 0
            indices = np.arange(size)
            if shuffle == True:
                np.random.shuffle(indices)
            X_copy = []
            for i in range(len(X)):
                X_copy.append(X[i][indices])
            y_copy = y[indices]
            continue

def batch_generator_multi_Y(X, Y, batch_size, shuffle=True):
    """
    Batch generator, X is a list contains inputs from different sources
    y is a numpy array
    """
    size = X[0].shape[0]
    indices = np.arange(size)
    if shuffle == True:
        np.random.shuffle(indices)
    X_copy = []
    Y_copy = []
    for i in range(len(X)):
        X_copy.append(X[i][indices])
    for i in range(len(Y)):
        Y_copy.append(Y[i][indices])
    i = 0
    while True:
        if i + batch_size < size:
            x_batch = []
            y_batch = []
            for j in range(len(X)):
                x_batch.append(X_copy[j][i:i + batch_size])
            for j in range(len(Y)):
                y_batch.append(Y_copy[j][i:i + batch_size])
            yield x_batch, y_batch
            i += batch_size
        else:
            x_batch = []
            y_batch = []
            for j in range(len(X)):
                x_batch.append(X_copy[j][i:size])
            for j in range(len(Y)):
                y_batch.append(Y_copy[j][i:size])
            yield x_batch, y_batch
            i = 0
            indices = np.arange(size)
            if shuffle == True:
                np.random.shuffle(indices)
            X_copy = []
            for i in range(len(X)):
                X_copy.append(X[i][indices])
            Y_copy = []
            for i in range(len(Y)):
                Y_copy.append(Y[i][indices])
            continue