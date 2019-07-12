import tensorflow as tf
import numpy as np
import os

'''

crnn:
1. build the model
2. train the model
3. fine tune the model
4. predict the model
triplet loss implementation for crnn speaker identification.
Set batch_size as None for testing purpose
'''


class crnn():
    '''
    crnn model, defined by dimension, length
    '''
    def __init__(self,
                 time,
                 dim,
                 attention_layer,
                 h_list1,
                 h_list2,
                 dropout = True,
                 num_class = 1211,
                 attention = True,
                 att_position = 'cnn_out',
                 att_activation = 'sigmoid',
                 squared = True):

        '''
        :param time: get time steps of feature
        :param dim: get dimension of feature
        :param h_list1: get rnn hidden shape 1
        :param h_list2:  get rnn hidden shape 2
        :param num_class: get number of classes if softmax is applied
        :param attention: if attention is applied or not
        :param att_position: where to apply attention possible value 'cnn_out','rnn_out'
        '''


        self.length = time
        self.dim = dim
        self.attention_layer = attention_layer
        self.epsilon = 1e-16
        self.num_class = num_class
        self.attention = attention
        self.att_activation = att_activation
        self.tensor = {}
        self.gradient = {}
        self.dropout = dropout
        '''self.embeddings = None
        self.pairwise_dist = None
        self.triplet_loss = None
        self.mask = None
        self.loss = None'''
        # for rnn
        self.hidden_list_1 = h_list1
        # get attention
        self.hidden_list_2 = h_list2
        self.att_position = att_position
        self.squared = squared

   # builder1: init params and return 
    def _initialize(self):
        '''
            initialize params;
        '''
        # Input
        # x with shape (batch_size, feature shape)
        # y with shape (batch_size,)
        self.X = tf.placeholder("float32", [None, self.length, self.dim])
        self.y = tf.placeholder("float32", [None])
        self.y_oh = tf.one_hot(tf.cast(self.y, tf.int32), depth=self.num_class)


    def conv_bn_pool(self, input, conv_filter, conv_strides, pool_shape, pool_strides):

        '''
        create conv-batchnorm-pooling combo
        :param input:
        :param conv_filter: tf tensor shape of (width, length, num_channel_in, num_channel_out)
        :param conv_strides: python list length of four
        :param pool_shape:
        :param pool_strides:
        :return:
        '''

        conv = tf.nn.conv2d(input, conv_filter, strides = conv_strides, padding='SAME')
        conv = tf.nn.relu(conv)
        conv = tf.nn.pool(conv, window_shape = pool_shape, strides = pool_strides, pooling_type = 'MAX', padding = 'SAME')
        #conv = tf.layers.batch_normalization(conv)
        return conv


    # builder2: define crnn skeleton
    def RNN(self,
            cnn_front_end = True,
            cnn_shape = [64,128],
            embedding_size = 1024):

        '''
        rnn model returns embeddings with range of all real numbers. A default embedding size is 512

        :param cnn_front_end:
        :param embedding_size:
        :return:
        '''

        if cnn_front_end:
            # exapnd 1 dimension for filters
            Input = tf.expand_dims(self.X, -1)

            # cnn front end
            with tf.variable_scope('cnn_r'):
                conv1_filter = tf.get_variable('conv1_filter', shape = [3, 3, 1, cnn_shape[0]], initializer = tf.contrib.layers.xavier_initializer())
                conv2_filter = tf.get_variable('conv2_filter', shape = [3, 3, cnn_shape[0], cnn_shape[1]], initializer = tf.contrib.layers.xavier_initializer())
                #conv3_filter = tf.get_variable('conv3_filter', shape = [3, 3, cnn_shape[1], cnn_shape[2]], initializer = tf.contrib)
                trans_output = self.conv_bn_pool(Input, conv1_filter, conv_strides = [1, 1, 1, 1], pool_shape = [2,2], pool_strides = [2,2])
                trans_output = self.conv_bn_pool(trans_output, conv2_filter, conv_strides = [1,1,1,1], pool_shape = [2,2], pool_strides = [2,2])
                f_shape = trans_output.get_shape().as_list()
                # concatenate feature maps of each time step together
                trans_output = tf.reshape(trans_output, [-1, f_shape[1], f_shape[2]*f_shape[3]])
        else:
            trans_output = self.X

        with tf.variable_scope('rnn'):
        # brnn
            fw_cell = [tf.nn.rnn_cell.GRUCell(n, kernel_initializer = tf.initializers.orthogonal()) for n in self.hidden_list_1]
            bw_cell = [tf.nn.rnn_cell.GRUCell(n, kernel_initializer = tf.initializers.orthogonal()) for n in self.hidden_list_1]

            fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell, state_is_tuple=True)


            bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell, state_is_tuple=True)
            if self.dropout:
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.5)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)

            # Get lstm cell output
            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                              bw_cell,
                                                              trans_output,
                                                              dtype=tf.float32,
                                                              time_major = False)

            # output layer
            outputs = tf.concat(outputs, 2)
            # collect tensors for gradient checking
            self.tensor['birnn_concat'] = outputs

        # if it taks self.X as input, then the attention is based on raw data frames.
        if self.attention:
            with tf.variable_scope('embedding_attention'):
                if self.att_position == 'cnn_output':
                    attention = self._attention_wrapper(trans_output)
                if self.att_position == 'rnn_output':
                    attention = self._attention_wrapper(outputs)
                output_shape = outputs.get_shape().as_list()
                # create attention tensor :[None, time, dim] copy attention vector 'dim' times
                attention = tf.tile(attention, [1, 1, output_shape[-1]], name = 'attention_distribution')
                # weighted sum embedding
                self.rnn_out = tf.reduce_sum(tf.multiply(attention, outputs), 1, name = 'rnn_output')


                '''embeddings = tf.layers.dense(self.rnn_out,
                                                  1024,
                                                  kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1))

                embeddings = tf.layers.dense(embeddings,
                                             512,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))'''

                embeddings = tf.layers.dense(self.rnn_out,
                                             embedding_size,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

                embeddings = self.l2_norm_helper(embeddings)


        else:
            with tf.variable_scope('embedding_laststep'):
                # outputs shape of [batch, timestep, 2*last_hidden_layer] reduce to => [batch, 2*last_hidden_layer]
                output = tf.reduce_mean(outputs, axis = 1)
                embeddings = tf.layers.dense(output,
                                                  embedding_size,
                                                  kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                                  kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 0.1))

                embeddings = self.l2_norm_helper(embeddings)

        self.embeddings = embeddings

    def l2_norm_helper(self, tensor):
        '''
        calculate tensor's l2 norm on axis 1.
        :param tensor: input tensor
        :return: l2 norm of the tensor
        '''
        return tensor/tf.expand_dims(tf.norm(tensor, ord = 2, axis = 1), -1)

    # builder3: attention model
    '''
    Needed find best
    '''

    def _attention_wrapper_dense(self, frames, shape):
        '''
         create mlp for attention model
         the attention model is initialized by random normal distribution with 0 mean and 1 std
        :param frames:
        :param shape: the shape of weight matrixes
        :return:
        '''

        attention_weights = tf.Variable(tf.random_normal(shape= shape, mean=0, stddev=1), name = 'attention_weights')
        # attention_b = tf.Variable(tf.constant(0.1), name = 'attention_weights')
        output = tf.matmul(frames, attention_weights)
        return output



    def _attention_wrapper(self, input_tensor):
        '''
        generate how much attention should be allocate to each step
        :param input_tensor: shape [None, time, dim]
        :return: tensor shape [None, time]
        '''
        input_shape = input_tensor.get_shape().as_list()
        # this layer map dim to a scalar without bias
        # tf.scan(lambda a,x: a+x)
        # frames in shape [batch*timestep, dimension)
        frames = tf.reshape(input_tensor, [-1, input_shape[2]])

        # denselayer wont take the undefined shape, customize dense is needed
        # linear activation is used

        attention_1 = self._attention_wrapper_dense(frames, shape = [input_shape[2], 1])
        if self.att_activation == 'tanh':
            attention_1 = tf.tanh(attention_1)
        if self.att_activation == 'sigmoid':
            attention_1 = tf.sigmoid(attention_1)
        if self.att_activation =='linear':
            attention_1 = attention_1

        attention_2 = tf.reshape(attention_1, [-1, input_shape[1], 1]) #batch, time, scalar
        attention_3 = tf.nn.softmax(attention_2, axis = 1)
        self.tensor['attention_vector'] = attention_3
            # softmax over time
        return attention_3


    def softmax_loss(self):
        with tf.variable_scope('softmax_loss'):
            self.logits = tf.layers.dense(self.embeddings, self.num_class, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)) # linear activation, loss is using softmax loss
            # create labels for softmax pretrain

            self.softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y_oh, logits = self.logits))

            for key, item in self.tensor.items():
                self.gradient[key] = tf.gradients(self.softmax_loss, item)
        return self.softmax_loss



    ##################################################################################################
    # Triplet Loss Functions
    ##################################################################################################
    # Distance Based
    def _pairwise_distance(self):

        '''
        Compute 2D matrix of distances
        calculate the squared distances in a batch
        :param embeddings: [batch, embedding_size]
        :return: squared distance matrix
        '''
        dot_product = tf.matmul(self.embeddings, tf.transpose(self.embeddings))
        square_norm = tf.diag_part(dot_product)
        # all distance are larger or equal to zero

        distance = tf.maximum(tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0), self.epsilon)
        if not self.squared:
            distance = tf.math.sqrt(distance)
        return distance


    # get positive triplet mask
    def _get_anchor_positive_triplet_mask(self):
        '''
        :return: 2d mask where mask[a,p] is True if a and p are distinct and have same label
        '''

        # Distinct Checking
        indices_equal = tf.cast(tf.eye(tf.shape(self.y)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        # positive checking
        labels_equal = tf.equal(tf.expand_dims(self.y, 0), tf.expand_dims(self.y, 1))
        mask = tf.logical_and(indices_not_equal, labels_equal)
        return mask

    # get negative triplet mask
    def _get_anchor_negative_triplet_mask(self):

        '''
        :return: 2d mask where mask[a,n] is True if a and n have distinct label
        '''

        labels_equal = tf.equal(tf.expand_dims(self.y, 0), tf.expand_dims(self.y, 1))
        mask = tf.logical_not(labels_equal)
        return mask

    # get valid triplet mask 3d
    def _get_triplet_mask(self):
        '''
         return a mask if Y[a] = Y[b], M[a,b] = 1, else 0
        :param Y:
        :return: 3-d matrix
        '''
        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(self.y)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(self.y, 0), tf.expand_dims(self.y, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)
        return mask

    def _restricted_loss(self, a_p_dist, a_n_dist, margin_plus):
        beta_1, beta_2 = margin_plus
        restricted_loss = tf.maximum(a_p_dist - beta_1, self.epsilon) - tf.minimum(a_n_dist - beta_2, self.epsilon)
        return restricted_loss

    # batch all triplet loss
    def batch_all_triplet_loss(self, margin_plus, margin):

        '''
         find the hardest triplet pairs of each elements in embeddings
         distance based triplet loss
         step1: get distance matrix
         step2: find positive matrix and negative
         step3: find hardest and generate loss
         according to label Y
         :param margin_plus: a tuple of positive margin and negative margin
        :param embeddings:
        :param Y:
        :return: anchor, hardest positive and hardest negative
        '''
        # get squared distance matrix

        pairwise_dist = self._pairwise_distance()
        self.pairwise_dist = pairwise_dist

        # adjust here
        a_p_dist = tf.expand_dims(pairwise_dist, 2)
        a_n_dist = tf.expand_dims(pairwise_dist, 1)

        # margin_plus: the first term activated when d_ap >= m1, second term activated when d_an <= m2
        if margin_plus:
            triplet_loss = self._restricted_loss(a_p_dist, a_n_dist, margin_plus)
        else:
            triplet_loss = tf.maximum(a_p_dist - a_n_dist + margin, self.epsilon)

        # get positive and negative matrix, return a mask if Y[a] = Y[b], M[a,b] = 1, else 0
        mask = self._get_triplet_mask()
        mask = tf.maximum(tf.to_float(mask), self.epsilon)
        self.mask = mask
        # element-wise multiplication
        # get
        triplet_loss = tf.multiply(mask, triplet_loss)
        # find loss >0
        valid_triplets = tf.to_float(tf.greater(triplet_loss, self.epsilon))
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets/ (num_valid_triplets + self.epsilon)
        # get final mean triplet loss over the positive valid triplets
        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + self.epsilon)
        return triplet_loss, num_positive_triplets


    # find hard triplet loss only
    def batch_hard_triplet_loss(self, margin_plus, margin):
        '''
        :param d: hinge triplet param
        :param margin:
        :return:
        '''
        pairwise_dist = self._pairwise_distance()

        # positive mask
        mask_anchor_positive = self._get_anchor_positive_triplet_mask()
        mask_anchor_positive = tf.to_float(mask_anchor_positive)
        anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)
        hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis = 1, keepdims = True)

        # negative mask
        mask_anchor_negative = self._get_anchor_negative_triplet_mask()
        mask_anchor_negative = tf.to_float(mask_anchor_negative)

        max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis = 1, keepdims = True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist *(1.0 - mask_anchor_negative)
        # shape (batch_size,)
        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis = 1, keepdims = True)

        # adjust triplet loss here###################################
        if margin_plus:
            margin_plus_1, margin_plus_2 = margin_plus
            triplet_loss_1 = tf.maximum(hardest_positive_dist - margin_plus_1, 0) - tf.minimum(hardest_negative_dist - margin_plus_2, 0)
        else:
            triplet_loss_1 = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, self.epsilon)
        triplet_loss_2 = tf.reduce_mean(triplet_loss_1)

        for key, item in self.tensor.items():
            self.gradient[key] = tf.gradients(triplet_loss_2, item)

        self.triplet_loss = triplet_loss_2

        return hardest_positive_dist, hardest_negative_dist, triplet_loss_2


    def batch_semi_hard(self):
        pass