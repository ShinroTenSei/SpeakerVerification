import os
import numpy as np
import tensorflow as tf
from .models_v2 import *
from .loss import *
from . import constants as c

class crnn_graph():
    '''
    build crnn computing graph
    '''
    def __init__(self, hyper_parameters):
        self.hp = hyper_parameters
        self.X = tf.placeholder("float32", [None, c.FB_FRAME_SIZE, c.N_FILT], name = 'X')
        self.y = tf.placeholder("float32", [None], name = 'y')

    def crnn(self):
        input_tensor = tf.expand_dims(self.X, -1)
        cnn_output = cnn(input_tensor, self.hp['cnn_shape'])
        rnn_output = rnn(cnn_output, self.hp['rnn_shape'], self.hp['rnn_type'])
        with tf.name_scope('embedding_attention'):
            if self.hp['attention']:
                weighted_sum = attention_wrapper(key = cnn_output,
                                                 value = rnn_output,
                                                 activation= self.hp['attention_activation'])

                embedding = tf.layers.dense(weighted_sum,
                                            self.hp['embedding_shape'],
                                            activation = tf.tanh,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1))
                '''embedding = tf.layers.dense(embedding,
                                            self.hp['embedding_shape'],
                                            activation=tf.tanh,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
                embedding = tf.layers.dense(embedding,
                                            self.hp['embedding_shape'],
                                            activation=tf.tanh,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
                embedding = tf.layers.dense(embedding,
                                            self.hp['embedding_shape'],
                                            activation=tf.tanh,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))'''
                self.embedding = tf.nn.l2_normalize(embedding, axis = -1, name = 'embedding')

            else:
                embedding = tf.reduce_mean(rnn_output, axis = 1)
                embedding = tf.layer.dense(embedding,
                                            self.hp['embedding_shape'],
                                            activation = tf.tanh,
                                            kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1))
                self.embedding = tf.nn.l2_normalize(embedding, axis = -1, name = 'embedding')


    def build_graph(self):
        self.crnn()
        with tf.name_scope('optimization'):
            self.global_step = tf.Variable(0)
            self.learning_rate = tf.placeholder('float32', name = 'lr')
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.loss = None
            if self.hp['strategy'] == 'restricted':
                self.loss = restricted_triplet_loss(self.embedding, self.y, self.hp['margin'])
            elif self.hp['strategy'] == 'semi_hard':
                self.loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(self.y, self.embedding, margin = self.hp['margin'])
                tf.add_to_collection('loss', self.loss)
            # clipped_gvs = [(tf.clip_by_global_norm(gvs, 3.0), var) for grad, var in gvs]
            self.gvs = self.optimizer.compute_gradients(self.loss)
            #tf.add_to_collection('gvs', self.gvs)
            self.train_op = self.optimizer.apply_gradients(self.gvs)
            tf.add_to_collection('train_op', self.train_op)
            self.saver = tf.train.Saver()
            self.init_g = tf.global_variables_initializer()
            self.init_l = tf.local_variables_initializer()


class att_dnn():
    def __init__(self, hp):
        self.hp = hp
        self.X = tf.placeholder("float32", [None, c.FB_FRAME_SIZE, c.N_FILT], name = 'X')
        self.y = tf.placeholder("float32", [None], name = 'y')





