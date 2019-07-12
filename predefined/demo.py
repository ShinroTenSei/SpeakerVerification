import os
import numpy as np
import tensorflow as tf
from . import constants as c
from .voice_input_module import *
from os.path import join

'''
this script defines demo object for demo_run

verifier = demo_verifier(model_path, audio_path)
verifier.enroll(.)
verifier.verify(.)
verifier.signout(.)


'''

class TestModel():
    def __init__(self, X, y, embedding):
        self.X = X
        self.y = y
        self.embeddings = embedding

def load_model(model_path, embedding_name = 'truediv'):
    tf.reset_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(join(model_path, 'model.meta'))
    saver.restore(self.sess, join(model_path, ',model'))
    X = sess.graph.get_tensor_by_name('Placeholder:0')
    y = sess.graph.get_tensor_by_name('Placeholder_1:0')
    embedding_tensor_name = 'embedding_attention/' + embedding_name + ':0'
    embedding = sess.graph.get_tensor_by_name(embedding_tensor_name)

    mdl = TestModel(X, y, embedding)
    return sess, mdl

def enroll():
    pass

def test():
    pass


