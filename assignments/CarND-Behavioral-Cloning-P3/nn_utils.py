import numpy as np
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def standardize_data(features, std_encoder, **kwargs):
    # ALT: normalize channel: [-1,1]
    tr_mean, tr_std = std_encoder
    features = features - tr_mean
    if kwargs['standardize']:
        features = features/tr_std
    if 'log' in kwargs and kwargs['log']:
        print("Dataset Mean: {:.2}, Std: {:.2}".format(features.mean(), features.std()) )
    return features

def preprocess_pipeline(features, target, std_encoder, **kwargs):
    # TBD: Resize images
    # Shuffle each partition
    features, target = shuffle(features, target)
    # Standardize RGB data (not greyscale) based on training fitted data
    features = standardize_data(features, std_encoder, **{'standardize': True, 'log': True})
    # ALT: steering angle normalized [-1,1]: corresponding to -25 to 25 degrees
    return features

def load_partitions(data_path):
    data_src = os.path.join(data_path, 'ckpts')
    if not os.path.exists(data_src): os.makedirs(data_src)

    with open(os.path.join(data_src, 'train.p'), mode='rb') as f:
        data_dict = pickle.load(f)
        X_train, y_train = (data_dict['image_files'], data_dict['angles'])
        del data_dict
    with open(os.path.join(data_src, 'validation.p'), mode='rb') as f:
        data_dict = pickle.load(f)
        X_val, y_val = (data_dict['image_files'], data_dict['angles'])
        del data_dict
    return X_train, y_train, X_val, y_val

def save_partitions(data_path, X_train, y_train, X_validation, y_validation):
    data_src = os.path.join(data_path, 'ckpts')
    if not os.path.exists(data_src): os.makedirs(data_src)

    with open(os.path.join(data_src, 'train.p'), 'wb') as f:
        train_dict = {'image_files':X_train, 'angles':y_train}
        pickle.dump(train_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_src, 'validation.p'), 'wb') as f:
        val_dict   = {'image_files':X_validation,   'angles':y_validation}
        pickle.dump(val_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

def partition_data(df_samples):
    X, y = df_samples['image'], df_samples['steering']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_val, y_val

def partition_samples(data_path, df_samples):
    # Train-test split (80/20) due to small number of training examples
    # split based on the filesnames, without actually loading the data, data processed and loaded via generator
    # Keras Image Generator does not shuffle the validation data nor shuffle data before split
    data_src = os.path.join(data_path, 'ckpts')
    if not os.path.exists(data_src): os.makedirs(data_src)

    X, y = df_samples['image'], df_samples['steering']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    with open(os.path.join(data_src, 'train.p'), 'wb') as f:
        train_dict = {'image_files':X_train, 'angles':y_train}
        pickle.dump(train_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_src, 'validation.p'), 'wb') as f:
        val_dict   = {'image_files':X_val,   'angles':y_val}
        pickle.dump(val_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return X_train, y_train, X_val, y_val
