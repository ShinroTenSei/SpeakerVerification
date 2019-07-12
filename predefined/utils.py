import os
import numpy as np
import tensorflow as tf
import pandas as pd
from .wav_reader import *
from .vox_cache_v2 import *
from .graph import *
import random
import pickle as pkl
import time
from time import gmtime, strftime
import shutil
import matplotlib.pyplot as plt
import math

'''

Initialization module. Since in this experiment, batch is generated online due to large data size(30gb filter banks cache).

'''

'''
an example of hyper parameter:
        hp = {
              'message':' Tester can be deleted dropout.',  # A message of the purpose of the research.
              'cnn_shape': [64, 128],
              'rnn_shape': [256, 256],
              'embedding_shape': 256,
              'strategy': 'semi_hard',
              'margin': (0.2, 0.8),         
              'attention':  True, # using attention mechanism or not
              'attention_activation': 'linear',
              'learning_rate': 1e-4,
              'epoch_num': 200,
              'num_enroll':5,
              'num_test': 10,
              'torlerance': 20,
              'batch_user':32,
              'batch_user_sample':8,
              'test_path':
              'num_enroll':5,
              'num_test:10,
             }
        
'''

'''
initialize evaluation paths.
'''




def init_evaluation(test_path, num_users, num_enroll, num_test):
    '''

    :param test_path:
    :param num_users:
    :param num_enroll:
    :param num_test:
    :return:
    '''


    print('Initializing testing set.')
    id_list = os.listdir(test_path)
    # list of dirs for evaluation
    enroll_list, test_list = [], []
    key = lambda x: os.path.getsize(x)
    if num_users:
        id_list = random.sample(id_list, num_users)

    for id in id_list:
        current_root = os.path.join(test_path, id)

        paths = []
        for root, dirs, files in os.walk(current_root):
            if len(files):
                paths.extend([os.path.join(root, f) for f in files if f.endswith('wav')])
        for _ in range(num_enroll):
            enroll_list.append(max([dir for dir in paths if dir not in enroll_list], key=key))
        # if number of test files is specified:
        if num_test and num_enroll + num_test < len(paths):
            test_list.extend([dir for dir in paths if dir not in enroll_list][:num_test])
        else:
            test_list.extend([dir for dir in paths if dir not in enroll_list])
    return enroll_list, test_list

def search_ops(txt):
    return [n.name for n in tf.get_default_graph().as_graph_def().node if txt in n.name]

class RestoreGraphVintage():
    # log_path, the path to save the model
    def __init__(self, log_path):
        tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph(join(log_path, 'model.meta'))
        saver.restore(sess, join(log_path, 'model'))
        self.X = sess.graph.get_tensor_by_name('Placeholder:0')
        self.y = sess.graph.get_tensor_by_name('Placeholder_1:0')
        # embedding = sess.graph.get_tensor_by_name('embedding:0')
        self.embedding = sess.graph.get_tensor_by_name('embedding_attention/truediv:0')
        self.sess = sess


class RestoreGraph():
    def __init__(self, log_path, hp):
        self.hp = hp
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(os.path.join(log_path, 'model.meta'))
        self.saver.restore(self.sess, os.path.join(log_path, 'model'))
        self.X = self.sess.graph.get_tensor_by_name('X:0')
        self.y = self.sess.graph.get_tensor_by_name('y:0')
        self.learning_rate = self.sess.graph.get_tensor_by_name('optimization/lr:0')
        self.relu = self.sess.graph.get_tensor_by_name('cnn/relu0:0')
        self.embedding = self.sess.graph.get_tensor_by_name('embedding_attention/embedding:0')
        if self.hp['strategy'] == 'restricted':
            self.loss = restricted_triplet_loss(self.embedding, self.y, self.hp['margin'])
        elif self.hp['strategy'] == 'semi_hard':
            self.loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(self.y, self.embedding, margin=self.hp['margin'])
            self.loss = tf.identity(self.loss, name="loss")
        self.gvs = tf.get_collection('gvs')
        self.train_op = tf.get_collection('train_op')[0]



# prepare for training
def init_main(hp, new_record = True, record_path = None):
    tf.reset_default_graph()
    print('Default graph resetting finished.')
    if record_path:
        hp = pkl.load(open(os.path.join(record_path, 'hp'),'rb'))
        train_log = pkl.load(open(os.path.join(record_path, 'train_log'), 'rb'))

    else:
        # create profile
        date_time = strftime("%Y-%m-%d-%H_%M:%S", gmtime())
        record_path = os.path.join(hp['model_path'], 'result' + date_time.split('_')[0])
        if new_record == True:
            # make dir and write basic info
            #-----------------------------------------------------
            if os.path.isdir(record_path):
                shutil.rmtree(record_path, ignore_errors = True)
            os.mkdir(record_path)
            print('Created new record: '+ record_path)


            with open(os.path.join(record_path, 'Readme.txt'), 'w+') as f:
                for k,v in hp.items():
                    f.write(k + ':' + str(v)+ '\n')
                f.close()

            pkl.dump(hp, open(os.path.join(record_path, 'hp'), 'wb'))
            #------------------------------------------------------
        # generate split train test
        user_list = os.listdir(hp['train_path'])
        dev_list = random.sample(user_list, hp['test_size'])
        train_list = [u for u in user_list if u not in dev_list]

        train_log = {
                    'epoch': 0,
                    'train_loss':[],
                    'test_loss':[],
                    'best_loss':float('inf'),
                    'record_path':record_path,
                    'train_list': train_list,
                    'test_list': dev_list
                    }

        if new_record == True:
            pkl.dump(train_log, open(os.path.join(record_path, 'train_log'), 'wb'))

    enroll_list, test_list = init_evaluation(hp['test_path'], 40, hp['num_enroll'], hp['num_test'])
    return train_log, enroll_list, test_list

'''
train utils
'''

def generate_batch_v2(path_to_cache,
                      progress,
                      batch_size,
                      num_sample):

    # 1. sample batch ids from current progress:
    batch_ids = random.sample(progress, batch_size)
    # update current progress
    progress = [id for id in progress if id not in batch_ids]

    #2. collect all data into a dictionary
    res_dictionary = {}
    for label in batch_ids:
        if label[-1] == ')':
            label = label[:-3]

        x = pkl.load(open(os.path.join(path_to_cache, label), 'rb'))
        res_dictionary[label] = x

    # 3. aggregate the result into list of batches(arrays)
    X_array_list = []
    y_array_list = []

    # until any entry in res_dict runs out:
    while True:

        cur_batch_x = []
        cur_batch_y = []
        # travers in the dictionary collect results from different users

        for k,v in res_dictionary.items():
            #pop num_sample from feature
            if len(v)>= num_sample:
                cur_batch_x = cur_batch_x + v[:num_sample]
                cur_batch_y.extend([k for _ in range(num_sample)])
                res_dictionary[k] = v[num_sample:]
            else:
                break

        if len(cur_batch_x) == num_sample*batch_size:
        # collect result
            X_array = np.stack(cur_batch_x)
            y_array = np.stack(cur_batch_y)
            X_array_list.append(X_array)
            y_array_list.append(y_array)
        else:
            break


    return X_array_list, y_array_list, progress

class train():
    def __init__(self,train_log, hp ):
        self.train_log = train_log
        self.hp = hp

    def init_test(self):
        self.test_Xs, self.test_ys, _ = generate_batch_v2(self.hp['train_path'],
                                                          self.train_log['test_list'],
                                                          self.hp['test_size'],
                                                          self.hp['batch_user_sample'])


    def epoch_train(self, sess, graph, learning_rate, randomization = True):

        # train for a epoch
        # initialize progress
        self.progress = self.train_log['train_list']
        epoch_loss = []
        while len(self.progress) > self.hp['batch_user']:

            Xs, ys, self.progress = generate_batch_v2(self.hp['train_path'], self.progress, self.hp['batch_user'], self.hp['batch_user_sample'])
            assert len(Xs) == len(ys)
            if randomization:
                index = random.sample(range(len(Xs)), 1)
                Xs = [Xs[i] for i in index]
                ys = [ys[i] for i in index]
            for X, y in zip(Xs, ys):
                _, loss = sess.run([graph.train_op, graph.loss], feed_dict = {graph.X:X, graph.y:y, graph.learning_rate:learning_rate})
                epoch_loss.append(loss)
        avg_loss = sum(epoch_loss)/float(len(epoch_loss))
        self.train_log['train_loss'].append(avg_loss)
        return avg_loss

    def epoch_test(self, sess, graph, learning_rate):
        dev_loss = []
        for X, y in zip(self.test_Xs, self.test_ys):
            loss = sess.run(graph.loss, feed_dict = {graph.X:X, graph.y:y, graph.learning_rate: learning_rate})
            dev_loss.append(loss)
        avg_loss = sum(dev_loss)/float(len(dev_loss))
        self.train_log['test_loss'].append(avg_loss)
        return avg_loss

    def call_back(self,sess, graph, test_loss):
        # call back based on test loss
        if test_loss <= self.train_log['best_loss']:
            print('Valid loss improved to ' + str(test_loss))
            self.train_log['best_loss'] = test_loss
            print('Train log updated!')
            if os.path.isdir(self.train_log['record_path']):
                graph.saver.save(sess, join(self.train_log['record_path'], 'model'))
                pkl.dump(self.train_log, open(join(self.train_log['record_path'], 'train_log'), 'wb'))
                print('Model saved!')

            return 0
        else:
            return 1


    def monitor(self, sess, graph):
        monitor_X, monitor_y = self.test_Xs[0], self.test_ys[0]
        gradient_monitor = graph.gvs[-3][0]
        relu_monitor = sess.graph.get_tensor_by_name('cnn/relu0:0')
        gvs, relu = sess.run([gradient_monitor, relu_monitor], feed_dict = {graph.X:monitor_X, graph.y:monitor_y})
        return gvs, relu


    def lr_search(self, lr_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3,5e-3, 1e-2, 5e-2, 1e-1]):
        self.init_test()
        train_loss_list = []
        test_loss_list = []
        for lr in lr_list:
            tf.reset_default_graph()
            new_graph = crnn_graph(self.hp)
            new_graph.build_graph()

            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                train_loss = self.epoch_train(sess,new_graph,lr, randomization = True)
                test_loss = self.epoch_test(sess,new_graph, lr)
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)

        return train_loss_list, test_loss_list

    def plot_lr_search(self, train_loss_list, test_loss_list):
        plt.plot(lr_list, train_loss_list, label='train loss')
        plt.plot(lr_list, test_loss_list, label='test loss')
        plt.set_xticks(lr_list)
        plt.set_xlabel('learning rate')
        plt.set_ylabel('loss')
        plt.legend()
        plt.show()

def train_func(t, graph):
    tor = 0
    start_learning_rate = t.hp['learning_rate']
    t.init_test()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(t.hp['epoch_num']):
        print('-----------------------------------------------------')
        print('Epoch ' + str(epoch) + ' training...')
        learning_rate = start_learning_rate * math.exp(-0.1 * epoch)
        start_time = time.time()
        train_loss = t.epoch_train(sess, graph, learning_rate)
        test_loss = t.epoch_test(sess, graph, learning_rate)
        v =  t.call_back(sess, graph, test_loss)
        if v:
            tor+=v
        else:
            tor = 0
        elapsed = round(time.time() - start_time, 2)
        print('Epoch took ' + str(elapsed) + ' seconds.')
        print('Epoch train loss: ' + str(round(train_loss, 10)) + '. Epoch test loss: '+ str(round(test_loss, 10)) +'.')
        if tor >= t.hp['tolerance']:
            print('-----------------------------------------------------')
            print('Tolerance violated. Finished Training')
            break

def substitute_txt(path, new_string, *args):
    '''
    this function substitute the line starts with certain character with a new line
    :return: None
    '''
    with open(path, 'r+') as f:
        lines = f.readlines()
    with open(path, 'w') as f:
        # adjust the file read pointer to the first line
        for i, line in enumerate(lines):
            if i == len(lines) - 1:
                line = line + ', ' + new_string
            f.write(line)
        f.close()

def plot_loss(train_log):
    plt.plot(train_log['train_loss'], label = 'train loss')
    plt.plot(train_log['test_loss'], label = 'test loss')
    plt.xticks(range(len(train_log['train_loss'])))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

