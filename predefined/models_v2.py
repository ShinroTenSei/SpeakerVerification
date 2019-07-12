import os
import numpy as np
import tensorflow as tf
from . import constants as c
'''
sequencial model with input params:
params = {'length':128 # int, time 
          'dim': 13 # int, dimension
          'num_class': 1251 # int, number of classes
          }


'''

def conv_bn_pool(tensor,
                 i,
                 conv_filter,
                 conv_strides,
                 pool_shape,
                 pool_strides):
    '''
    create conv-batchnorm-pooling combo
    :param input:
    :param conv_filter: tf tensor shape of (width, length, num_channel_in, num_channel_out)
    :param conv_strides: python list length of four
    :param pool_shape:
    :param pool_strides:
    :return:
    '''
    conv = tf.nn.conv2d(tensor, conv_filter, strides=conv_strides, padding='SAME')
    conv = tf.layers.batch_normalization(conv)
    conv = tf.nn.leaky_relu(conv, name = 'relu'+ str(i))
    conv = tf.nn.pool(conv, window_shape=pool_shape, strides=pool_strides, pooling_type='MAX', padding='SAME')
    return conv

# cnn front end for crnn model
def cnn(tensor, cnn_shape, initializer = tf.contrib.layers.xavier_initializer()):
    with tf.name_scope('cnn'):
        i = 0
        # the first conv layer uses [1,1] pooling stride
        conv_filter = tf.get_variable('conv_' + str(i) + '_filter', shape = [3, 3, 1, cnn_shape[i]], initializer = initializer)
        tensor = conv_bn_pool(tensor, i, conv_filter, conv_strides = [1, 1, 1, 1], pool_shape = [3,3], pool_strides = [1,1])
        i = 1
        conv_filter = tf.get_variable('conv_' + str(i) + '_filter', shape = [3, 3, cnn_shape[i-1], cnn_shape[i]], initializer = initializer)
        tensor = conv_bn_pool(tensor, i, conv_filter, conv_strides = [1, 1, 1, 1], pool_shape = [3,3], pool_strides = [2,2])
        f_shape = tensor.get_shape().as_list()
        # concatenate feature maps of each time step together
        # reshape the output as (None, time, dims*channels) where time is shifted by conv layers and channel equals to cnn_shape[-1]
        cnn_output = tf.reshape(tensor, [-1, f_shape[1], f_shape[2]*f_shape[3]], name = 'cnn_output')
    return cnn_output

# rnn back end for crnn model
def rnn(input_tensor, hidden_list, rnn_type, initializer = tf.orthogonal_initializer()):
    with tf.name_scope('rnn'):
        if rnn_type == 'gru':
            fw_cell = [tf.nn.rnn_cell.GRUCell(n, kernel_initializer= initializer) for n in hidden_list]
            bw_cell = [tf.nn.rnn_cell.GRUCell(n, kernel_initializer= initializer) for n in hidden_list]
        elif rnn_type == 'lstm':
            fw_cell = [tf.nn.rnn_cell.LSTMCell(n, initializer= initializer) for n in hidden_list]
            bw_cell = [tf.nn.rnn_cell.LSTMCell(n, initializer= initializer) for n in hidden_list]

        fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                          bw_cell,
                                                          input_tensor,
                                                          dtype = tf.float32,
                                                          time_major = False)

        outputs = tf.concat(outputs, 2, name='rnn_output')

    return outputs



def dnn(input_tensor, dim):
    with tf.name_scope('dnn'):
        outputs = tf.layers.dense(input_tensor, dim, activation = tf.nn.relu)
        outputs = tf.layers.dense(outputs, dim, activation = tf.nn.relu)
        outputs = tf.layers.dense(outputs, dim, activation = tf.nn.relu)
    return outputs

def get_attention_distribution(input_tensor, input_shape, activation = 'linear'):
    frames = tf.reshape(input_tensor, [-1, input_shape[2]])
    attention_model_weights = tf.get_variable('attention_weights',
                                        shape = [input_shape[2], 1],
                                        initializer = tf.contrib.layers.xavier_initializer())
    attention_ln_output = tf.matmul(frames,attention_model_weights)
    attention_activated = attention_ln_output
    if activation == 'tanh':
        attention_activated = tf.tanh(attention_ln_output)
    if activation == 'sigmoid':
        attention_activated = tf.sigmoid(attention_ln_output)
    attention_reshaped = tf.reshape(attention_activated, [-1, input_shape[1], 1])
    attention_distribution = tf.nn.softmax(attention_reshaped, axis = 1, name = 'attention_distribution')
    return attention_distribution

def attention_wrapper(key, value, activation = 'linear'):
    key_shape = key.get_shape().as_list()
    value_shape = value.get_shape().as_list()
    attention_distribution = get_attention_distribution(key, key_shape, activation = activation)
    attention_tiled = tf.tile(attention_distribution, [1,1,value_shape[-1]])
    weighted_sum = tf.reduce_sum(tf.multiply(attention_tiled, value), 1, name = 'weighted_sum')
    return weighted_sum


def get_tensors(graph=tf.get_default_graph()):
    return [t for op in graph.get_operations() for t in op.values()]

