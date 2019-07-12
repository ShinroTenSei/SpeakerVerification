import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from collections import Counter
from scipy import stats
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from .wav_reader import *

'''
Evaluate based on knn probabilities.
'''

class knn_eval():
    def __init__(self,
                 enroll_paths,
                 test_paths,
                 output_path,
                 type='fb',
                 cached = False,
                 segment_based = False):
        '''
        usage example:
        E = eval(enroll_list, test_list, output_path,mdl, sess)
        records = E.inference()
        result, predictions, ground_truth = E.evaluate()
        E.output_embeddings(predictions, ground_truth)
        '''


        self.enroll_paths = enroll_paths
        self.test_paths = test_paths
        self.output_path = output_path
        self.type = type
        self.records = None
        self.cached = cached
        self.segment_based = segment_based

    def _get_emb(self, path, mdl, sess):
        '''
         returns utterance level embedding.
        :param path:
        :return:
        '''
        if self.cached:
            features = pkl.load(open(path, 'rb'))
        else:
            if self.type == 'spectrogram':
                features = get_fft_spectrum(path)
            if self.type == 'fb':
                features = get_filterbank(path)
            if self.type == 'mfcc':
                features = get_mfcc(path)
        embeddings = sess.run(mdl.embedding, feed_dict={mdl.X: features})
        # remove outliers from embeddings
        # embeddings = self.rm_outlier(embeddings)
        return list(embeddings)

    def get_space(self, mdl, sess, space='enroll'):
        '''
         get predictions or enrollments
        :param space:
        :return:
        '''
        if space == 'enroll':
            paths = self.enroll_paths
        elif space == 'test':
            paths = self.test_paths
        embedding_vectors = []
        labels = []
        for path in paths:
            id = [p for p in path.split('/') if p.startswith('id')][0]
            val = self._get_emb(path, mdl, sess)
            if self.segment_based:
                l = len(val)
                embedding_vectors.extend(val)
                labels.extend([id for _ in range(l)])
            else:
                val = np.mean(np.stack(val), axis = 0)
                embedding_vectors.extend(val)
                labels.append(id)

        return np.stack(embedding_vectors), np.stack(labels)

    def get_records(self, mdl, sess):
        # enrollments: list length of n enroll segments length of 200ms
        enrollments, enroll_ids = self.get_space(mdl, sess, space='enroll')
        # predictions: list length of n predictions
        predictions, ground_truth = self.get_space(mdl, sess, space='test')

        records = {
            'enroll_ids': enroll_ids,
            'enrollments': enrollments,
            'ground_truth': ground_truth,
            'predictions': predictions
        }

        return records

    def get_knn(self, records, n_neighbors=10, leaf_size=30):
        # n nearest depends on average number of enrollments
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
        knn.fit(records['enrollments'], records['enroll_ids'])
        # knn.predict(records['predictions'])
        return knn


    def evaluate_knn(self, knn, records):
        # probabilities with shape(n_samples, n_enrolls) to sample and enrollment correspondingly
        probas = knn.predict_proba(records['predictions'])
        score = knn.score(records['predictions'], records['ground_truth'])
        unique_sorted_ids = sorted(list(set(list(records['enroll_ids']))))
        fpr_list = []
        fnr_list = []
        eer_list = []
        for i, label in enumerate(unique_sorted_ids):
            fpr, tpr, thres = roc_curve(records['ground_truth'], probas[:, i], pos_label=label)
            fnr = 1 - tpr
            fpr_list.append(fpr)
            fnr_list.append(fnr)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            # thresh = interp1d(fpr, thres)(eer)
            eer_list.append(eer)
        eer_array = np.array(eer_list)
        eer = round(eer_array.mean(), 4)
        return eer, score, fpr_list, fnr_list

    def evaluate_graph(self, mdl, sess, n_neighbors=30, leaf_size=30):
        records = self.get_records(mdl, sess)
        knn = self.get_knn(records, n_neighbors=n_neighbors, leaf_size=leaf_size)
        eer, score, fpr_list, fnr_list = self.evaluate_knn(knn, records)

        result = {'eer': eer,
                  'fpr': fpr_list,
                  'fnr': fnr_list}

        print("EER on test set: " + '{0:.2f}%'.format(eer * 100))
        return result, records, score


    def output_embedding(self, records):
        embeddings = records['predictions']
        labels = records['ground_truth']
        pd.DataFrame(embeddings).to_csv(os.path.join(self.output_path, 'embeddings'), sep='\t', index=False,header=False)
        pd.DataFrame(labels).to_csv(os.path.join(self.output_path, 'labels'), sep='\t', index=False, header=False)
        print('embeddings output complete!')





'''
evaluate based on negative distance measurements
'''
class distance_eval():
    def __init__(self,
                 enroll_paths,
                 test_paths,
                 output_path,
                 type='fb',
                 cached=True,
                 segment_based = False):

        self.enroll_paths = enroll_paths
        self.test_paths = test_paths
        self.output_path = output_path
        self.type = type
        self.records = None
        self.cached = cached
        self.segment_based = segment_based

    def _get_emb(self, path, mdl, sess):
        '''
        inference for one sample
        :param path:
        :param mdl:
        :param sess:
        :return:
        '''
        if self.cached:
            features = pkl.load(open(path, 'rb'))
        else:
            if self.type == 'spectrogram':
                features = get_fft_spectrum(path)
            if self.type == 'fb':
                features = get_filterbank(path)
            if self.type == 'mfcc':
                features = get_mfcc(path)
        embeddings = sess.run(mdl.embedding, feed_dict={mdl.X: features})
        # remove outliers from embeddings
        # embeddings = self.rm_outlier(embeddings)
        return list(embeddings)

    def get_space(self, mdl, sess, space='enroll'):
        '''
         get predictions or enrollments
        :param space:
        :return:
        '''
        if space == 'enroll':
            paths = self.enroll_paths
            embedding_dict = {}

            for path in paths:
                id = [p for p in path.split('/') if p.startswith('id')][0]
                val = self._get_emb(path, mdl, sess)
                # apply emb of all 2-seconds segments
                if id in embedding_dict.keys():
                    embedding_dict[id].extend(val)
                else:
                    embedding_dict[id] = val
            embedding_vectors = []
            labels = []
            # calculate user_vector
            for k, v in embedding_dict.items():
                labels.append(k)
                embedding_vectors.append(np.mean(np.stack(v), axis=0))

        elif space == 'test':
            paths = self.test_paths
            embedding_vectors = []
            labels = []
            # for each test audio
            for path in paths:
                id = [p for p in path.split('/') if p.startswith('id')][0]
                val = self._get_emb(path, mdl, sess)

                embedding_vectors.extend(val)
                labels.extend([id for _ in range(len(val))])
            assert len(embedding_vectors) == len(labels)
        return np.stack(embedding_vectors), np.stack(labels)

    def get_records(self, mdl, sess):
        # enrollments: list length of n enroll segments length of 200ms
        enrollments, enroll_ids = self.get_space(mdl, sess, space='enroll')

        # predictions: list length of n predictions
        predictions, ground_truth = self.get_space(mdl, sess, space='test')
        records = {
            'enroll_ids': enroll_ids,
            'enrollments': enrollments,
            'ground_truth': ground_truth,
            'predictions': predictions
        }
        return records

    def evaluate_distance(self,records):
        '''
        :param debug:
        :return:
        '''
        enroll_ids = records['enroll_ids']
        enrollments = records['enrollments']
        ground_truth = records['ground_truth']
        predictions = records['predictions']
        assert len(enrollments.shape) == 2
        assert len(predictions.shape) == 2

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
        thresholds_list = []
        print('Evaluating each id.')

        # get eer for each label in enroll ids
        for i, label in enumerate(enroll_ids):
            # distance between enrollment mean and test audio of the ith user.
            cur_dist = dist_matrix[:, i]
            # generate similarity score based on euclidean distance
            score = -cur_dist
            if np.unique(score).shape == (1,):
                raise Exception('Suspicious distance found in ' + str(i) + 'th column.')
            fpr, tpr, thresholds = roc_curve(ground_truth, score, pos_label=label, drop_intermediate = False)

            fnr = 1 - tpr
            # update min_len if needed and trim fpr, fnr arrays with min_len
            # eer is the minimum(fnr,fpr) when smallest difference between fpr and fnr
            fpr_list.append(fpr)
            fnr_list.append(fnr)
            thresholds_list.append(thresholds)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1)
            eer_list.append(eer)


        eer_array = np.array(eer_list)
        eer = round(eer_array.mean(), 4)

        result = {'eer': eer,
                  'fpr': fpr_list,
                  'fnr': fnr_list,
                  'eer_list':eer_list,
                  'thres_list':thresholds_list}
        self.result = result
        print("EER on test set: " + '{0:.2f}%'.format(eer * 100))
        return result

    def predict(self, records, mdl, sess, path):
        embeddings = self._get_emb(path, mdl, sess)
        utterance_emb = np.mean(np.stack(embeddings), axis = 0, keepdims = True)
        dist_matrix = euclidean_distances(utterance_emb, records['enrollments'])

        prediction = records['enroll_ids'][np.argmin(dist_matrix)]
        return prediction, dist_matrix

    def evaluate_graph(self, mdl, sess):
        records = self.get_records(mdl, sess)
        result = self.evaluate_distance(records)
        return result, records

