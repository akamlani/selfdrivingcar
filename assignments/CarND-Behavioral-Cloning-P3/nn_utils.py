import numpy as np
import pandas as pd
import cv2

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
    return features





# steering angle normalized [-1,1]: corresponding to -25 to 25 degrees
