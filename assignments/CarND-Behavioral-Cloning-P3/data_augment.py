import numpy as np
import pandas as pd
import cv2
import os

from functools import reduce

import data_utils as du
import nn_utils as nu
import cv_transf as cvt


def aug_image(fname, steer, fn, **kwargs):
    fname_fn, steer_fn = cvt.augment_image(fname, steer, fn, **kwargs)
    return (fname_fn, steer_fn, fn.__name__)

def aug_sample(fname, steer):
    # flip image horizontal
    data_samples = []
    data_samples.append((fname, steer, 'original'))
    data_samples.append( aug_image(fname, steer, cvt.gaussian_blur_image) )
    params = {'axis': 'vertical', 'adjust_target': True}
    data_samples.append( aug_image(fname, steer, cvt.flip_image, **params) )
    # params = {'min': 2.5, 'max': 10, 'adjust_target': True}
    # data_samples.append( aug_image(fname, steer, cvt.rotate_image, **params) )
    # params = {'width_range': 0.10, 'height_range': 0.05, 'adjust_target': True}
    # data_samples.append( aug_image(fname, steer, cvt.shift_image, **params) )
    params = {'min': 0.25, 'max': 0.6}
    data_samples.append( aug_image(fname, steer, cvt.gamma_corr_image, **params) )
    params = {'min': 0.25,'max': 0.5}
    data_samples.append( aug_image(fname, steer, cvt.brightness_image, **params) )
    data_samples.append( aug_image(fname, steer, cvt.hist_equalize_image) )
    return data_samples

def augment_training(X_train, y_train):
    df_train = pd.DataFrame([X_train, y_train]).T
    df_train = df_train.apply(lambda x: aug_sample(x['image'], x['steering']), axis=1)
    df_train = df_train.reset_index(drop=True)
    df_train = pd.DataFrame(reduce(lambda x,y: x+y, df_train), columns=['image', 'steering', 'augment'])
    df_train = df_train[['image', 'steering']]
    return (df_train['image'], df_train['steering'])

def get_track_data(training_path, sim_path, track):
    df_collection = pd.DataFrame()
    track_path = os.path.join(data_training, track)
    for collection_path in sim_path:
        data_collection_path = os.path.join(track_path, collection_path)
        df_sim = du.reformat_csv(data_collection_path,  header=False)
        df_collection = df_collection.append(df_sim)
        print('Track: {}, Collections: {}, Num Collection Samples: {}'.format(track, collection_path, len(df_sim)) )
    print('Track: {}, Collections: {}, Num Collections Samples: {}'.format(track, sim_path, len(df_collection)) )
    return df_collection

def create_track_samples(training_path, sim_path):
    df_sim_track1    = get_track_data(data_training, sim_path, 'track1')
    #df_sim_track2    = get_track_data(data_training, sim_path, 'track2')
    df_sim_comb      = df_sim_track1#pd.concat([df_sim_track1, df_sim_track2], axis=0)
    df_drive_sim     = df_sim_comb.sample(frac=1.0).reset_index(drop=True)
    # use left, right angles of camera and shift accordingly: steering already in float format
    df_sim_shift     = du.lateral_shift(df_drive_sim)
    df_drive_samples = du.combine_dataset(df_sim_shift)
    print( "Num Simulator Samples w/L,C,R: {}".format(len(df_drive_samples)) )
    return df_drive_samples


if __name__ == '__main__':
    #opt = 'udacity'
    opt = 'training_sim'

    if opt == 'udacity':
        # load and reformat/clean csv file
        data_udacity     = 'data/udacity/'
        data_path        = data_udacity
        df_drive_udacity = du.reformat_csv(data_udacity, header=True)
        # use left, right angles of camera and shift accordingly: steering already in float format
        df_udacity_shift = du.lateral_shift(df_drive_udacity)
        df_udacity_samples = du.combine_dataset(df_udacity_shift)
        print("Udacity Num Overall Samples: {}".format(len(df_drive_samples)) )
        X_train, y_train, X_val, y_val = nu.partition_data(df_udacity_samples)
        print("Udacity Number Train Obs: {}, Validation Obs: {}".format(len(X_train), len(X_val)) )
        # perform augmentation on training set
        X_train, y_train = augment_training(X_train, y_train)
        print("Udacity w/Augmented Train Obs: {}, Validation Obs: {}".format(len(X_train), len(X_val)))

    else:
        # Get Track 1+2 Centered Augmented Data w/L,R camera shifts
        data_training    = 'data/training/'
        data_path        = data_training
        df_base_samples  = create_track_samples(data_training, ['training.centered/'])
        X_train, y_train, X_val, y_val = nu.partition_data(df_base_samples)

        print( "Number Train Obs: {}, Validation Obs: {}".format(len(X_train), len(X_val)) )
        X_train, y_train = augment_training(X_train, y_train)
        print("W/Augmented Train Obs: {}, Validation Obs: {}".format(len(X_train), len(X_val)))
        # add in recovery and curves wo/augmentation to both training and validation, still using L,R cameras
        recover_path = ['training.curves/']#['training.curves/', 'training.recover/']
        df_recover_samples  = create_track_samples(data_training, recover_path)
        X_train_rec, y_train_rec, X_val_rec, y_val_rec = nu.partition_data(df_recover_samples)
        # combine both for a final training/validation set
        X_train = X_train.append(X_train_rec)
        y_train = y_train.append(y_train_rec)
        X_val   = X_val.append(X_val_rec)
        y_val   = y_val.append(y_val_rec)
        print("Number Train Obs: {}, Validation Obs: {}".format( (len(X_train),len(y_train)), (len(X_val),len(y_val)) ))


    # serialize final partitions to disk, based on option
    nu.save_partitions(data_path, X_train, y_train, X_val, y_val)
    print('Saved Partitions to path: {}'.format(data_path))


    """
    data_path = data_training
    df_drive  = df_drive_sim
    # use left, right angles of camera and shift accordingly: steering already in float format
    df_sim_shift = du.lateral_shift(df_drive)
    # partition into train/validation split, augmenting only training data
    df_drive_samples = du.combine_dataset(df_sim_shift)
    print( "Num Overall Samples: {}".format(len(df_drive_samples)) )
    X_train, y_train, X_val, y_val = nu.partition_samples(data_path, df_drive_samples)
    print( "Number Train Obs: {}, Validation Obs: {}".format(len(X_train), len(X_val)) )
    # perform augmentation on training set
    X_train, y_train = augment_training(X_train, y_train)
    print("W/Augmented Train Obs: {}, Validation Obs: {}".format(len(X_train), len(X_val)))
    # seralize partitions to disk
    nu.save_partitions(data_path, X_train, y_train, X_val, y_val)
    """
