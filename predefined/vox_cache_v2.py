import platform
import sys
from os.path import join
import os

if platform.system() == "Windows":
    sys.path.append('.\\')
sys.path.append(".\\")

import pickle as pkl
from .wav_reader import *
import random

'''
cachier utils
'''

def get_user_catalog(sample_user, path_to_ids):
    user_catalog = {}
    # get data catalog
    for user in sample_user:
        # if it is an id:
        if user.startswith('id'):
            label = int(user.strip('id'))
        # else skip
        else:
            continue

        user_root = os.path.join(path_to_ids, user)
        cur_paths = []

        # traverse the user file tree, aggregate all wav dirs.
        for root, dirs, files in os.walk(user_root):
            if len(files):
                cur_paths.extend([os.path.join(root,f) for f in files if f.endswith('wav')])
        user_catalog[str(label)] = cur_paths
    return user_catalog

def extract_and_dump(user_catalog,
                     enroll_num,
                     path_to_train_cache,
                     f_type):
    '''
    extract and dump feature for each user
    :param user_catalog:
    :return:
    '''

    for label, dirs in user_catalog.items():
        '''
        for each element in dumping_list
        shape is [time, numcep]
        '''
        if enroll_num and enroll_num < len(dirs):
            sampled_dir = random.sample(dirs, enroll_num)
        else:
            sampled_dir = dirs

        dumping_list = []
        for dir in sampled_dir:
            res = None
            if f_type == 'spectrogram':
                res = get_fft_spectrum(dir)
            if f_type == 'filterbank':
                res = get_filterbank(dir)
            if f_type == 'mfcc':
                res = get_mfcc(dir)
            if res is not None:
                dumping_list.extend(list(res))
        fname = os.path.join(path_to_train_cache, str(label))

        dump(dumping_list, fname)




def dump(dumping_list, filename):
    '''

    X_lst: python lst of arrays
    y_lst: nparray of labels
    max_len: max seq lenth of current batch
    filename: cache name
    '''
    try:
        os.remove(filename)
    except OSError:
        pass

    with open(filename, 'wb') as file:
        pkl.dump(dumping_list, file)
        file.close()

'''
training data cachier
'''

def cachier(path_to_ids,
            path_to_train_cache,
            f_type,
            user_num = None,
            enroll_num = None,
            catalog = None):

    # check and make dirs
    if not os.path.isdir(path_to_train_cache):
        os.mkdir(path_to_train_cache)

    user_catalog = None
    # if catalog is passed in
    if catalog:
        user_catalog = catalog

    else:
        # get all user ids
        user_list = os.listdir(path_to_ids)
        if user_num:
            # sample user and get wav dirs:
            sample_user = random.sample(user_list, user_num)
        else:
            sample_user = user_list

        # collect all wav dirs w.r.t each user label
        user_catalog = get_user_catalog(sample_user, path_to_ids)

    if user_catalog:
        # extract and dumping:
        extract_and_dump(user_catalog,
                         enroll_num,
                         path_to_train_cache,
                         f_type)

        return user_catalog

    else:
        return None


'''
test data cachier(utterance-based)
default feature type: filterbank
'''

def test_cache(path_to_ids, cache_path, f_type='filterbank'):
    '''
    generate cache based on testing wavs,
    :param
    path_to_ids: directory to wav folder of test set

    '''
    # check and make dirs
    if not os.path.isdir(cache_path):
        os.mkdir(cache_path)

    user_list = os.listdir(path_to_ids)
    user_catalog = get_user_catalog(user_list, path_to_ids)

    for label, dirs in user_catalog.items():
        if not os.path.isdir(join(cache_path, label)):
            os.mkdir(join(cache_path, label))
        label_path = join(cache_path, label)
        for i, dir in enumerate(dirs, 1):
            res = None
            if f_type == 'spectrograms':
                res = get_fft_spectrum(dir)
            if f_type == 'filterbank':
                res = get_filterbank(dir)
            if f_type == 'mfcc':
                res = get_mfcc(dir)
            # dump uttrance features with shape(num_frames, time_steps, dimensions)
            if res is not None:
                file_dir = join(label_path, str(i))
                with open(file_dir, 'wb') as file:
                    pkl.dump(res, file)
                    file.close()


