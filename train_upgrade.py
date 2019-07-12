# __author__ ='shengtanwu'
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import matplotlib as plt
import os
import math
import random

from predefined.models import *
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm,trange,tqdm_notebook

'''
file running command:
python evaluation.py
'''

def parse_args():
    '''
    parse parameters
    :return: args
    '''
    parser = argparse.ArgumentParser(description = 'Data cache name and directory')
    parser.add_argument('-dir', '--directory', type = str, help = 'directory of data cache file: path/to/caches')
    parser.add_argument('-rdir','--restore_dir', type = str, default = './ckpt', help= 'directory to save model')
    #parser.add_argument('-dist', '--distance', type = str, help = 'distance measurements.')
    parser.add_argument('-lr', '--learning_rate', type = float, default = 0.1, help = 'learning rate of the model')
    parser.add_argument('-lmd', '--lambda', type = float, default = 0.2, help = 'regularization parameter')
    #parser.add_argument('-ds', '--display_step', type = int, default = 5, help = 'display every # of steps')
    parser.add_argument('-epc','--epochs',type = int, default = 100, help = 'training steps')
    parser.add_argument('-mg', '--margin', type = float, default = 1.0, help = 'margin')
    parser.add_argument('-bs', '--batch_size', type = float, default = 128, help = 'batch size')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    print(tf.__version__)
    gym = gym(args)
    gym.load_data()
    gym.build()
    loss = gym.train(epoches = args.epochs)




'''
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

nums_classes = y_train.shape[1]
height = x_train.shape[2]
width = x_train.shape[1]
hidden_list = [512,256,128]

# Inputshape init
X = tf.placeholder("float32", [None, width, height])
Y = tf.placeholder("float32", [None, num_classes])

weights = {'out':tf.Variable(tf.random_normal([hidden_list[-1]*2, num_classes])*0.1)}
bias = {'out':tf.Variable(tf.zeros([1, num_classes])+0.1)}
'''

