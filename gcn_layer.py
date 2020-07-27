import numpy as np
from scipy.sparse.linalg import eigs
import tensorflow as tf

def Scaled_Laplacian(W):
    W = W.astype(float)
    n = np.shape(W)[0]
    d = []
    #simple graph, W_{i,i} = 0
    L = -W
    #get degree matrix d and Laplacian matrix L
    for i in range(n):
        d.append(np.sum(W[i, :]))
        L[i, i] = d[i]
    #symmetric normalized Laplacian L
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])

    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    # lambda_max \approx 2.0
    # we can replace this sentence by setting lambda_max = 2
    return np.matrix(2 * L / lambda_max - np.identity(n))

def Cheb_Poly(L, Ks):
    assert L.shape[0] == L.shape[1]
    n = L.shape[0]
    L0 = np.matrix(np.identity(n))
    L1 = np.matrix(np.copy(L))
    L_list = [np.copy(L0), np.copy(L1)]
    for i in range(1, Ks):
        Ln = np.matrix(2 * L * L1 - L0)
        L_list.append(np.copy(Ln))
        L0 = np.matrix(np.copy(L1))
        L1 = np.matrix(np.copy(Ln))
    # L_lsit (Ks, n*n), Lk (n, Ks*n)
    return np.concatenate(L_list, axis=-1)

def First_Approx(W, n):
    #first order approximation
    A = W + np.identity(n)
    d = []
    for i in range(n):
        d.append(np.sum(A[i, :]))
    sinvD = np.sqrt(np.matrix(np.diag(d)).I)
    return np.identity(n) + sinvD * A * sinvD

def graph_conv(inputs, supports, dim_in, dim_out, scope='gcn',
               initializer = tf.contrib.layers.xavier_initializer()):
    #inputs: shape is [batch, num_nodes, features]
    #supports: [num_nodes, num_nodes*(order+1)], calculate the chebyshev polynomials in advance to save time
    dtype = inputs.dtype
    num_nodes = inputs.get_shape().as_list()[1]
    assert num_nodes == supports.shape[0]
    assert dim_in == inputs.shape[2]
    #in fact order is order-1
    order = int(supports.shape[1] / num_nodes)
    x_new = tf.reshape(tf.transpose(inputs, [0, 2, 1]), [-1, num_nodes])    #[batch*feature, num_nodes]
    x_new = tf.reshape(tf.matmul(x_new, supports), [-1, dim_in, order, num_nodes])
    x_new = tf.transpose(x_new, [0, 3, 1, 2])                        #[batch, num_nodes, dim_in, order]
    x_new = tf.reshape(x_new, [-1, order*dim_in])
    with tf.variable_scope(scope):
        weights = tf.get_variable('weights', [order*dim_in, dim_out], dtype=dtype,
                                  initializer=initializer)
        #tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(weights))
        biases = tf.get_variable('biases', [dim_out], dtype=dtype,
                                 initializer = tf.constant_initializer(0.0, dtype=dtype))
        outputs = tf.nn.bias_add(tf.matmul(x_new, weights), biases)
    return tf.reshape(outputs, [-1, num_nodes, dim_out])


