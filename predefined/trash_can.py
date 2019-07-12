# __author__ ='shengtanwu'
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import matplotlib as plt
import tensorflow as tf
import os
import math
from os.path import isfile, isdir, join
from sklearn.model_selection import train_test_split
from keras import models
from keras.models import load_model
from keras.layers import *
from keras import regularizers

'''
aborted evaluate function from utils.eval
'''
def evaluate(records,
             debug=False):
    '''
    :param debug:
    :return:
    '''
    enroll_ids = records['enroll_ids']
    enrollments = records['enrollments']
    ground_truth = records['ground_truth']
    predictions = records['predictions']

    print('Start calculating distance matrix.')
    # get distance matrix,shape num_predictions * num_enrollments
    dist_matrix = euclidean_distances(predictions, enrollments, squared=True)
    assert dist_matrix.shape == (len(ground_truth), len(enroll_ids))
    assert len(list(np.where(dist_matrix == 1e-16)[0])) == 0
    assert np.all(dist_matrix > 0)
    print('Distance matrix calculation finished.')
    # for each class, calculate roc-auc statistics, generate a eer and average all to get the final
    eer_list = []
    fpr_list = []
    fnr_list = []
    print('Evaluating each id.')

    # get eer for each label in enroll ids
    for i, label in enumerate(enroll_ids):
        # distance between enrollment mean and test audio of the ith user.
        cur_dist = dist_matrix[:, i]
        # generate similarity score based on euclidean distance
        score = dist_score(cur_dist)
        if np.unique(score).shape == (1,):
            raise Exception('Suspicious distance found in ' + str(i) + 'th column.')
        fpr, tpr, _ = roc_curve(ground_truth, score, pos_label=label)
        fnr = 1 - tpr
        # update min_len if needed and trim fpr, fnr arrays with min_len
        # eer is the minimum(fnr,fpr) when smallest difference between fpr and fnr
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        eer_index = np.argmin(abs(fpr - fnr))
        eer = (fpr[eer_index]+fnr[eer_index])/2.0
        eer_list.append(eer)
    eer_array = np.array(eer_list)
    eer = round(eer_array.mean(), 4)

    # get average fpr and fnr list for plotting
    min1 = min([a.shape[0] for a in fpr_list])
    min2 = min([a.shape[0] for a in fnr_list])
    min_len = min(min1, min2)
    fpr_list = [a[:min_len] for a in fpr_list]
    fnr_list = [a[:min_len] for a in fnr_list]

    if debug:
        print(min_len)
        print(set([a.shape[0] for a in fpr_list]), set([a.shape[0] for a in fnr_list]))

    assert len(set([a.shape[0] for a in fpr_list])) == 1
    assert len(set([a.shape[0] for a in fnr_list])) == 1

    fpr = np.mean(np.array(fpr_list), axis=0)
    fnr = np.mean(np.array(fnr_list), axis=0)

    result = {'eer': eer,
              'fpr': fpr,
              'fnr': fnr}

    print("EER on test set: " + '{0:.2f}%'.format(eer * 100))
    return result, predictions, ground_truth