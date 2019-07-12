import os
import numpy as np
import tensorflow as tf
import argparse
import sys
from predefined.demo import *
from predefined.utils import *
from predefined.evaluation import *
from predefined.vox_cache_v2 import *

def parse_args():
    '''
    parse parameters
    :return: args
    '''

    parser = argparse.ArgumentParser(description = 'Parse Args to initialize prediction')
    parser.add_argument('-enroll', '--enrollment_dir', type = str, default = '../')
    parser.add_argument('-test', '--test_dir', type = str, default = '../')
    parser.add_argument('-mdl', '--log_path', type = str, default = '../models/att_restricted_batchall')
    args = parser.parse_args()
    return args


def run_demo():
    args = parse_args()
    graph = RestoreGraphVintage(args.log_path)
    sess = graph.sess
    enroll_paths = []
    test_paths = []
    for root, dirs, files in os.walk(args.enrollment_dir):
        if len(files):
            enroll_paths.extend([os.path.join(root, f) for f in files if f.endswith('wav')])

    for root, dirs, files in os.walk(args.test_dir):
        if len(files):
            test_paths.extend([os.path.join(root, f) for f in files if f.endswith('wav')])

    E = distance_eval(enroll_paths = enroll_paths,
                      test_paths = test_paths,
                      output_path = args.log_path,
                      cached = False)

    records = E.get_records(graph, sess)
    score = 0
    # for each test files
    for n in range(len(test_paths)):
        # get audio file embedding
        emb = np.mean(np.stack(E._get_emb(test_paths[n], graph, sess)), axis=0, keepdims=True)
        # get ground truth label from test path
        truth = [t for t in test_paths[n].split('/') if t.startswith('id')][0]
        # get prediction from enrollment records
        pred = records['enroll_ids'][np.argmin(euclidean_distances(np.squeeze(emb).reshape(1, -1), records['enrollments']))]
        if truth == pred:
            score += 1
        print('Truth: ' + [t for t in test_paths[n].split('/') if t.startswith('id')][0] + '. Predict: ' + pred)
    print('Totally ' + str(len(test_paths)) + ' test utterances. ' + str(score) + ' predictions are correct.')


if __name__ == '__main__':
    run_demo()