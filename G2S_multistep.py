from gcn_layer import *
import tensorflow as tf
from lib.metrics import MAE, RMSE, MAPE, MARE, R2

def Conv_ST(inputs, supports, kt, dim_in, dim_out, activation):
    '''
    :param inputs: a tensor of shape [B, T, N, C]
    :param supports:
    :param kt: temporal convolution length
    :param dim_in:
    :param dim_out:
    :return:
    '''
    T = inputs.get_shape().as_list()[1]
    num_nodes = inputs.get_shape().as_list()[2]
    assert inputs.get_shape().as_list()[3] == dim_in
    if (dim_in > dim_out):
        w_input = tf.get_variable(
            'wt_input', shape=[1, 1, dim_in, dim_out], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        res_input = tf.nn.conv2d(inputs, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif (dim_in < dim_out):
        res_input = tf.concat(
            [inputs, tf.zeros([tf.shape(inputs)[0], T, num_nodes, dim_out - dim_in])], axis=3)
    else:
        res_input = inputs
    # padding zero
    padding = tf.zeros([tf.shape(inputs)[0], kt - 1, num_nodes, dim_in])
    # extract spatial-temporal relationships at the same time
    inputs = tf.concat([padding, inputs], axis=1)
    x_input = tf.stack([inputs[:, i:i + kt, :, :] for i in range(0, T)], axis=1)    #[B*T, kt, N, C]
    x_input = tf.reshape(x_input, [-1, kt, num_nodes, dim_in])
    x_input = tf.transpose(x_input, [0, 2, 1, 3])

    if (activation == 'GLU'):
        conv_out = graph_conv(tf.reshape(x_input, [-1, num_nodes, kt * dim_in]),
                              supports, kt * dim_in, 2 * dim_out)
        conv_out = tf.reshape(conv_out, [-1, T, num_nodes, 2 * dim_out])
        out = (conv_out[:, :, :, 0:dim_out] + res_input) * \
              tf.nn.sigmoid(conv_out[:, :, :, dim_out:2 * dim_out])
    if (activation == 'sigmoid'):
        conv_out = graph_conv(tf.reshape(x_input, [-1, num_nodes, kt * dim_in]),
                              supports, kt * dim_in, dim_out)
        out = tf.reshape(conv_out, [-1, T, num_nodes, dim_out])
    # out = tf.nn.relu(conv_out + res_input)
    return out

def LN(y0, scope):
    # batch norm
    size_list = y0.get_shape().as_list()
    T, N, C = size_list[1], size_list[2], size_list[3]
    mu, sigma = tf.nn.moments(y0, axes=[1, 2, 3], keep_dims=True)
    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, T, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, T, N, C]))
        y0 = (y0 - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return y0

def attention_t(query, values, scope):
    '''
        :param query: a tensor shaped [B, Et]
        :param values: a tensor shaped [B, T, H*W, F]
        :return:
    '''
    Et = query.get_shape().as_list()[1]
    T = values.get_shape().as_list()[1]
    N = values.get_shape().as_list()[2]
    F = values.get_shape().as_list()[3]
    values_in = tf.reshape(values, [-1, T, N*F])  #[B, T, N*F]
    values_in = tf.transpose(values_in, [0, 2, 1]) #[B, N*F, T]
    values = tf.transpose(values_in, [2, 0, 1])  # [T,B,N*F]
    with tf.variable_scope(scope):
        Wv = tf.get_variable('Wv', shape=[T, N*F,1], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        bias_v = tf.get_variable('bias_v', initializer=tf.zeros([T]))
        Wq = tf.get_variable('Wq', shape=[Et, T], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
    value_linear = tf.reshape(tf.transpose(tf.matmul(values, Wv), [1,0,2]), [-1, T])
    #score = tf.nn.tanh((value_linear + bias_v) + tf.matmul(query, Wq))
    score = tf.nn.tanh((value_linear + bias_v) + tf.matmul(query, Wq))
    score = tf.nn.softmax(score, dim=1)  # shape is [B,T]
    values = tf.matmul(values_in, tf.expand_dims(score, axis=-1))  # [B,N*F,1]
    values = tf.reshape(tf.transpose(values, [0, 2, 1]), [-1, 1, N, F])
    return values

def attention_c(query, values, scope):
    '''
    :param query: a tensor shaped [B, Et]
    :param values: a tensor shaped [B, 1, H*W, F]
    :return:
    '''
    Et = query.get_shape().as_list()[1]
    N = values.get_shape().as_list()[2]
    F = values.get_shape().as_list()[3]
    values_in = tf.reshape(values, [-1, N, F])
    values = tf.transpose(values_in, [2, 0, 1]) #[F,B,N]
    with tf.variable_scope(scope):
        Wv = tf.get_variable('Wv', shape=[F, N,1], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())  #[F,N,1]
        bias_v = tf.get_variable('bias_v', initializer=tf.zeros([F]))
        Wq = tf.get_variable('Wq', shape=[Et, F], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
    value_linear = tf.reshape(tf.transpose(tf.matmul(values, Wv), [1, 0,2]), (-1, F))
    score = tf.nn.tanh((value_linear + bias_v) + tf.matmul(query, Wq))
    score = tf.nn.softmax(score, dim=1) #shape is [B,F]
    values = tf.matmul(values_in, tf.expand_dims(score,axis=-1)) #[B,N,1]
    return values

class Graph(object):
    def __init__(self, adj_mx, params, is_training):
        # self.adj_mx = adj_mx
        self.supports = np.float32(Cheb_Poly(Scaled_Laplacian(adj_mx), 2))
        self.params = params
        C, O = params.closeness_sequence_length, params.nb_flow
        H, W, = params.map_height, params.map_width
        Et, Em = params.et_dim, params.em_dim
        Horizon = params.horizon
        self.c_inp = tf.placeholder(tf.float32, [None, C, H, W, O], name='c_inp')
        inputs = tf.reshape(self.c_inp, [-1, C, H * W, O])  # [batch, seq_len, num_nodes, dim]
        self.et_inp = tf.placeholder(tf.float32, (None, Horizon, Et), name='et_inp')
        self.labels = tf.placeholder(tf.float32, shape=[None, Horizon, H, W, O], name='label')
        labels = tf.reshape(self.labels, (-1, Horizon, H * W, O))

        #long term encoder, encoding 1 to 12
        with tf.variable_scope('block1'):
            l_inputs = Conv_ST(inputs, self.supports, kt=3, dim_in=O, dim_out=32, activation ='GLU')
            l_inputs = LN(l_inputs, 'ln1')
        with tf.variable_scope('block2'):
            l_inputs = Conv_ST(l_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
            l_inputs = LN(l_inputs, 'ln2')
        with tf.variable_scope('block3'):
            l_inputs = Conv_ST(l_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
            l_inputs = LN(l_inputs, 'ln3')
        with tf.variable_scope('block4'):
            l_inputs = Conv_ST(l_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
            l_inputs = LN(l_inputs, 'ln4')
        with tf.variable_scope('block5'):
            l_inputs = Conv_ST(l_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
            l_inputs = LN(l_inputs, 'ln5')
        with tf.variable_scope('block6'):
            l_inputs = Conv_ST(l_inputs, self.supports, kt=2, dim_in=32, dim_out=32, activation='GLU')
            l_inputs = LN(l_inputs, 'ln6')

        #short term encoder, working differently for training and testing
        preds = []
        window = 3
        if is_training == True:
            label_padding = inputs[:, -window:, :, :]
            padded_labels = tf.concat((label_padding, labels), axis=1)
            print(padded_labels.shape)
            padded_labels = tf.stack([padded_labels[:, i:i + window, :, :] for i in range(0, Horizon)], axis=1)
            print('shape of padded labels:', padded_labels.shape)  # [B, Horizon, window, H*W, O]
            for i in range(0, Horizon):
                s_inputs = padded_labels[:, i, :, :, :]  #[B, window, N, O]
                et_inp = self.et_inp[:, i, :]
                with tf.variable_scope('horizon'+str(i)):
                    with tf.variable_scope('block7'):
                        gs_inputs = Conv_ST(s_inputs, self.supports, kt=3, dim_in=O, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln7')
                    '''
                    with tf.variable_scope('block8'):
                        gs_inputs = Conv_ST(gs_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln8')
                    '''
                    with tf.variable_scope('block9'):
                        gs_inputs = Conv_ST(gs_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln9')
                    ls_inputs = tf.concat((gs_inputs, l_inputs), axis=1)
                    print(ls_inputs.shape)
                    ls_inputs = attention_t(et_inp, ls_inputs, 'attn_t')
                    if params.nb_flow == 1:
                        pred = attention_c(et_inp, ls_inputs, 'dim1')
                    if params.nb_flow == 2:
                        pred = tf.concat((attention_c(et_inp, ls_inputs, 'dim1'),
                                          attention_c(et_inp, ls_inputs, 'dim2')), axis=-1)
                preds.append(pred)
        else:
            label_padding = inputs[:, -window:, :, :]
            for i in range(0, Horizon):
                s_inputs = label_padding
                et_inp = self.et_inp[:, i, :]
                with tf.variable_scope('horizon' + str(i)):
                    with tf.variable_scope('block7'):
                        gs_inputs = Conv_ST(s_inputs, self.supports, kt=3, dim_in=O, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln7')
                    '''
                    with tf.variable_scope('block8'):
                        gs_inputs = Conv_ST(gs_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln8')
                    '''
                    with tf.variable_scope('block9'):
                        gs_inputs = Conv_ST(gs_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln9')
                    ls_inputs = tf.concat((gs_inputs, l_inputs), axis=1)
                    print(ls_inputs.shape)
                    ls_inputs = attention_t(et_inp, ls_inputs, 'attn_t')
                    if params.nb_flow == 1:
                        pred = attention_c(et_inp, ls_inputs, 'dim1')
                    if params.nb_flow == 2:
                        pred = tf.concat((attention_c(et_inp, ls_inputs, 'dim1'),
                                          attention_c(et_inp, ls_inputs, 'dim2')), axis=-1)
                label_padding = tf.concat((label_padding[:, 1:,:,:], tf.expand_dims(pred, 1)), axis=1)
                preds.append(pred)

        self.preds = tf.stack(preds, axis=1)

        first_pred = preds[0]
        first_label = labels[:, 0, :, :]
        first_loss = tf.nn.l2_loss(first_pred - first_label)

        self.loss = tf.nn.l2_loss(self.preds - labels)
        #self.loss = tf.nn.l2_loss(self.preds - labels) + first_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=params.lr, beta1=params.beta1, beta2=params.beta2,
                                                epsilon=params.epsilon).minimize(self.loss)
        self.mean_rmse = RMSE(self.preds, labels) * params.scaler

        self.mae = []
        self.rmse = []
        self.mape = []
        self.r2 = []
        trues = tf.unstack(labels, axis=1)
        for _, (i, j) in enumerate(zip(preds, trues)):
            mae = MAE(i, j) * params.scaler
            self.mae.append(mae)
            rmse = RMSE(i, j) * params.scaler
            self.rmse.append(rmse)
            mape = MAPE(i, j, params.scaler, mask_value=10)
            self.mape.append(mape)
            r2 = R2(i, j)
            self.r2.append(r2)





