import numpy as np
import pandas as pd
import time
import pickle
import argparse
import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

import data_utils as du
import cv_features as cvf
import viz_utils as viz
import model as mdl
import search as sch
import f2f


from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


### Pipelines
def pipeline_tune_params(clf, ds_vehicles, ds_nonvehicles, n_samples):
    """
    acquire optimal tuning parameters for training sequence
    tuning parameters basded on K-Fold CV ROC/AUC Score
    """
    # search optimal space for features parameters
    config = {
        'color_space':  'YCrCb',      # can be RGB, HLS, HSV, LUV, YUV, YCrCb
        'spatial_size': (32,32),      # Spatial binning idmensions: downsample image (64,64)->(16,16)
        'hist_bins':    32,           # Number of Histogram Bins
        'orient': 9,                  # 6-9 according to HOG literature (9 may be better choice)
        'pix_per_cell': 8,
        'cells_per_block': 2,         # Block size to capture bigger features (helps w/normalization)
        'hog_channel': 'ALL',         # 'ALL' or 0,1,2 ('ALL') should provide better results
        'vis': False,
        'log': False
    }
    param_grid = {
        'color_space':        ['RGB', 'HLS', 'HSV', 'YUV', 'YCrCb'],
        'spatial_size':       [(16,16),(32,32),(48,48),(64,64)],
        'hist_bins':          [16,32,48,64],
        'orient':             [6,7,8,9],
        'pix_per_cell':       [4,8,12,16],
        'cells_per_block':    [2, 4],
        'hog_channel':        [0,1,2,'ALL']
    }

    tune_config_dict, tune_config = \
    mdl.fine_tune_features(clf, ds_vehicles, ds_nonvehicles, n_samples, config, param_grid)
    with open('./ckpts/tuned_params.p', 'wb') as f:
        pickle.dump({'params': tune_config, 'stats': tune_config_dict}, f)
    return tune_config_dict, tune_config


def pipeline_train(clf, **config):
    """
    Begin Training Sequence
    """
    X_train, y_train, X_val, y_val = \
    mdl.partition_data_basic(ds_vehicles, ds_nonvehicles, n_samples=-1, split_size=0.8, **config)
    scaler, X_train, X_val = mdl.scale_data(X_train, X_val)
    print('Training Split Distribution: {}'.format(dict(pd.Series(y_train).value_counts())) )
    print('Validation Split Distribution: {}'.format(dict(pd.Series(y_val).value_counts())) )

    # fit and serialize the model
    train_params = {'log': True, 'cv': True}
    clf, score = mdl.train(clf, X_train, y_train, X_val, y_val, **train_params)
    with open('./ckpts/models.p', 'wb') as f:
        pickle.dump({'model': clf, 'scaler': scaler}, f)
    with open('./ckpts/data_partition.p', 'wb') as f:
        pickle.dump({'train': (X_train, y_train), 'validation': (X_val, y_val)}, f)
    return clf, scaler


def pipeline_process_frame(base_img, scales=None, **kwargs):
    """
    axi: (ax1,ax2) axis is used for visualization of images
    """
    bbox_coords_l = []
    heatmap_coords_l = []
    for scale in scales:
        bbox_config = search_config.copy()
        bbox_config['scale'] = scale
        bbox_coords, heatmap_coords = find_coords(base_img, **bbox_config)
        for it in bbox_coords: bbox_coords_l.append(it)
        for it in heatmap_coords: heatmap_coords_l.append(it)
    # before optimization
    bbox_img = viz.draw_bboxes(base_img, bbox_coords_l)
    # minimize windows per heatmap
    draw_img, heat_thresh, labels = tracker.process_frame(base_img, heatmap_coords_l)
    heat_thresh = np.clip(heat_thresh, 0, 255)
    # plot images
    if 'axi' in kwargs and kwargs['axi']:
        ax1, ax2, ax3 = kwargs['axi']
        ax1.imshow(bbox_img)
        viz.set_axi_opts(ax1, **{'title': 'image: {}'.format(kwargs['name'])})
        ax2.imshow(draw_img)
        viz.set_axi_opts(ax2, **{'title': 'image: {}'.format(kwargs['name'])})
        ax3.imshow(heat_thresh, cmap='hot')
        viz.set_axi_opts(ax3, **{'title': 'heatmap thresh instances: {}'.format(labels[1])})
    return draw_img


### Utility functions
def find_coords(base_img, **config):
    """
    Get the coordinates for the original bounding boxes and heatmaps
    """
    finder.update_config(**config)
    tr_img, nsteps, hog_features = finder.config_hog_search(base_img)
    bbox_coords, heatmap_coords  = finder.subsample_hog_features(tr_img, clf, nsteps, hog_features)
    # bbox_img = viz.draw_boxes(base_img, bbox_coords)
    return bbox_coords, heatmap_coords





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vehicle Detection')
    parser.add_argument('-v', '--video', action='store_true', help='video processing')
    parser.add_argument('-t', '--train', action='store_true', help='train classifier')
    args = parser.parse_args()

    # acquire data
    ds_vehicles, ds_nonvehicles, df = du.get_data_full()
    # acquire best features if not already tuned
    if not os.path.exists('ckpts/tuned_params.p'):
        n_samples = int(len(ds_vehicles)*0.3)
        tune_config_dict, tune_config = pipeline_tune_params(LinearSVC(), ds_vehicles, ds_nonvehicles, n_samples)
        # update per optimal configuration
        tune_config_md = tune_config.copy()
        tune_config_md['hist_bins'] = 32
        tune_config_md['spatial_size'] = (32,32)
        tune_config_md['orient'] = 9
        with open('./ckpts/tuned_params.p', 'wb') as f:
            pickle.dump({'params': tune_config_md, 'stats': tune_config_dict}, f)

    with open('./ckpts/tuned_params.p', 'rb') as f:
        data_dict = pickle.load(f)
        tune_config = data_dict['params']
        tune_config_dict = data_dict['stats']
        del data_dict

    # train model
    if args.train:
        print("Begin Training Classifier")
        clf_svm = svm.SVC(kernel='linear', probability=True)
        clf, clf_scaler = pipeline_train(clf_svm, **tune_config)
    # extract serialed model from disk
    with open('./ckpts/models.p', 'rb') as f:
        data_dict = pickle.load(f)
        clf  = data_dict['model']
        clf_scaler = data_dict['scaler']
        del data_dict


    # define search configuration
    search_config = tune_config.copy()
    search_params = {
        'ys': [400,656],
        'xs': [200, None],
        'cells_per_step': 2,
        'scale': 1.5,
        'window': 64
    }
    search_config.update(search_params)

    window_size = 1 if not args.video else 20
    test_fnames = glob.glob('test_images/*')
    shape_dim = mpimg.imread(test_fnames[0]).shape
    finder  = sch.Search(clf_scaler, **search_config)
    tracker = f2f.F2FTracker(shape_dim[:2], window_size=window_size)
    scales  = [1.0, 1.25, 1.50, 2.0]

    if args.video:
        fname_input = 'videos/project_video.mp4'
        #fname_input = 'videos/test_video.mp4'
        fname, ext     = fname_input.split('.')
        fname_output   = "_".join([fname, 'tracked']) + "".join(['.', ext])
        print('Video Input: {}, Output: {}'.format(fname_input, fname_output))

        clip = VideoFileClip(fname_input)
        test_clip = clip.fl_image(lambda x: pipeline_process_frame(x, scales))
        test_clip.write_videofile(fname_output, audio=False)
    else:
        fig, ax = plt.subplots(3,3, figsize=(12,4))
        ax = ax.flatten()
        ax = [(ax[it],ax[it+1],ax[it+2]) for it in range(0,len(ax),3) ]
        [ pipeline_process_frame(mpimg.imread(fname), scales, **{'axi':axi, 'name': fname})
          for axi, fname in zip(ax,test_fnames[:3]) ]
        plt.tight_layout()
        fig.savefig('./output_images/heatmap_explore_1.png', transparent=False, bbox_inches='tight')
        plt.show()
