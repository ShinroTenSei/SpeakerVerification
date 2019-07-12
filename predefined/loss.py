import os
import numpy as np
import tensorflow as tf
from . import constants as c

##################################################################################################
# Triplet Loss Functions
##################################################################################################
# Distance Based
def _pairwise_distance(embeddings):
    '''
    Compute 2D matrix of distances
    calculate the squared distances in a batch
    :param embeddings: [batch, embedding_size]
    :return: squared distance matrix
    '''

    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)
    # all distance are larger or equal to zero

    distance = tf.maximum(tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0),
                           c.EPSILON, name='matrix_distance')    
    distance = tf.math.sqrt(distance)
    return distance


# get positive triplet mask
def _get_anchor_positive_triplet_mask(y):
    '''
    :return: 2d mask where mask[a,p] is True if a and p are distinct and have same label
    '''

    # Distinct Checking
    indices_equal = tf.cast(tf.eye(tf.shape(y)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    # positive checking
    labels_equal = tf.equal(tf.expand_dims(y, 0), tf.expand_dims(y, 1))
    mask = tf.logical_and(indices_not_equal, labels_equal)
    return mask


# get negative triplet mask
def _get_anchor_negative_triplet_mask(y):
    '''
    :return: 2d mask where mask[a,n] is True if a and n have distinct label
    '''

    labels_equal = tf.equal(tf.expand_dims(y, 0), tf.expand_dims(y, 1))
    mask = tf.logical_not(labels_equal)
    return mask


# get valid triplet mask 3d
def _get_triplet_mask(y):
    '''
     return a mask if Y[a] = Y[b], M[a,b] = 1, else 0
    :param Y:
    :return: 3-d matrix
    '''
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(y)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(y, 0), tf.expand_dims(y, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask



def restricted_triplet_loss(embeddings, y, margin_plus):
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
    pairwise_dist = _pairwise_distance(embeddings)
    # adjust here
    a_p_dist = tf.expand_dims(pairwise_dist, 2)
    a_n_dist = tf.expand_dims(pairwise_dist, 1)
    # margin_plus: the first term activated when d_ap >= m1, second term activated when d_an <= m2

    beta_1, beta_2 = margin_plus
    triplet_loss = tf.maximum(a_p_dist - beta_1, c.EPSILON) + tf.maximum(beta_2 - a_n_dist, c.EPSILON)
    # get positive and negative matrix, return a mask if Y[a] = Y[b], M[a,b] = 1, else 0
    mask = _get_triplet_mask(y)
    mask = tf.maximum(tf.to_float(mask), c.EPSILON)
    # element-wise multiplication
    # get
    triplet_loss = tf.multiply(mask, triplet_loss)
    # find where loss >0
    valid_triplets = tf.to_float(tf.greater(triplet_loss, c.EPSILON))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    # get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.truediv(tf.reduce_sum(triplet_loss), (num_positive_triplets + c.EPSILON), name='loss')
    return triplet_loss

