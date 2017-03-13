import numpy as np
import pandas as pd
import time

import matplotlib.image as mpimg

# local modules
import cv_utils as cvu
import cv_features as cvf
import cv_windows as cvw
import viz_utils as viz

# modeling
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score

def sample_features(ds, n_samples, **params):
    """
    Extract a sample of the features
    """
    log = params['log']
    params_config = params.copy()
    del params_config['log']

    random_idxs  = np.random.randint(0, len(ds), n_samples)
    ds_samples   = np.array(ds)[random_idxs]
    t_start  = time.time()
    features = cvf.extract_features(ds_samples, **params_config)
    t_end    = time.time()

    if log:  print('Feature Extraction Time: {}'.format(round(t_end-t_start, 2)) )
    return ds_samples, features

def partition_data_basic(ds_vehicles, ds_nonvehicles, n_samples=1000, split_size=0.8, **params):
    """
    Partition the dataset into train/validation partions
    TBD: Batch Sizes depending on classifier
    """
    split_size = 1 - split_size
    # extract features
    n_samples_vehicles = int(n_samples if n_samples != -1 else len(ds_vehicles))
    vehicle_samples, vehicle_features = sample_features(ds_vehicles, n_samples_vehicles, **params)
    n_samples_nonvehicles = int(n_samples if n_samples != -1 else len(ds_nonvehicles))
    nonvehicle_samples, nonvehicle_features = sample_features(ds_nonvehicles, n_samples_nonvehicles, **params)
    # Feature vector to represent a row in array; y=(1 for cars, 0 for noncars)
    X = np.vstack((vehicle_features, nonvehicle_features)).astype(np.float64)
    y = np.hstack( ((np.ones(len(vehicle_features))), np.zeros(len(nonvehicle_features))) )
    # shuffle data to avoid ordering
    X, y = shuffle(X, y)
    # create train/test splits via shuffle
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_size)
    if params['log']:
        print('Num Samples: Vehicles: {}, Nonvehicles: {}'.format(n_samples_vehicles, n_samples_nonvehicles))
        print('Num Samples: Train: {}, Validation: {}'.format(len(X_train), len(X_val)))
        print('Training Feature Vector Length: {}'.format(len(X_train[0])) )
    return X_train, y_train, X_val, y_val

def create_partitioned_features(X_train, X_val, n_samples=1000, **params):
    """
    Extract Features for train and validation partitions
    Input are the filenames already partitioned
    """
    n_samples_tr = n_samples if n_samples != -1 else len(X_train)
    tr_samples, tr_features = sample_features(X_train, n_samples_tr, **params)
    n_samples_val = n_samples if n_samples != -1 else len(X_val)
    val_samples, val_features = sample_features(X_val, n_samples_val, **params)
    s = 'Feature Vector Length Training: {}, Validation: {}'
    print(s.format(len(tr_features[0]), len(val_features[0])) )
    return tr_features, val_features

def scale_data(X_train, X_val):
    """
    Standard Scale the Training and Validation splits
    """
    std = StandardScaler().fit(X_train)
    X_train_sc = std.transform(X_train)
    X_val_sc   = std.transform(X_val)
    return std, X_train_sc, X_val_sc

def train(clf, X_train, y_train, X_val, y_val, **params):
    """
    Train and Evaluate the model
    Trained on *.png format
    """
    t_start = time.time()
    clf.fit(X_train, y_train)
    t_stop = time.time()
    # Check Score based on metric
    if 'cv' in params and params['cv']:
        score = lambda X, y: np.mean(cross_val_score(clf, X, y, cv=KFold(n_splits=5), scoring='roc_auc', n_jobs=-1))
    else:
        score = lambda X, y: round(clf.score(X, y), 4)
    if params['log']:
        print('Training Time: {}'.format(round(t_stop-t_start, 2)) )
        print('Training Accuracy of clf: {}: {}'.format(clf.__class__.__name__,  score(X_train, y_train)))
        print('Test Accuracy of clf: {}, {}'.format(clf.__class__.__name__, score(X_val, y_val)))
    return clf, score(X_val, y_val)

def predict(clf, features, n_samples, acc_opt=False):
    """
    Perform Prediction on a provided set of features and compare against metric (accuracy)
    """
    yhat = clf.predict(features[0:n_samples])
    if acc_opt:
        acc = (prediction == target[0:n_samples]).sum()/len(target[0:n_samples])
        print("Prediction Sample Accuracy: {}".format(acc) )
    return yhat


#### Tuning
def model_scoring(clf, ds_vehicles, ds_nonvehicles, n_samples, tune_config):
    """
    For a particular model configuration, partition/train/score the model
    """
    X_train, y_train, X_val, y_val = \
    partition_data_basic(ds_vehicles, ds_nonvehicles, n_samples, split_size=0.8, **tune_config)
    scaler, X_train, X_val = scale_data(X_train, X_val)
    clf, score = train(clf, X_train, y_train, X_val, y_val, **{'log': False, 'cv': True})
    return clf, score

def feature_tune(clf, ds_vehicles, ds_nonvehicles, n_samples, param_config, param_grid, param):
    """
    Score a particular parameter feature for evaluation
    """
    scoring = []
    for v in param_grid[param]:
        tune_config = param_config.copy()
        tune_config[param] = v
        clf, score = model_scoring(clf, ds_vehicles, ds_nonvehicles, n_samples, tune_config)
        data_dict = {param: v, 'tune_param': param, 'clf': clf.__class__.__name__, 'score': score}
        scoring.append(data_dict)
    return scoring

def update_optimal_tune(df_tune, config):
    """
    Update the new tuning configuration based on the most optimal score
    """
    max_score = max(df_tune.score)
    tune_param = df_tune.ix[df_tune.score.argmax()]['tune_param']
    param_config = df_tune.ix[df_tune.score.argmax()][tune_param]
    tune_config = config.copy()
    tune_config[tune_param] = param_config
    print( "Tuned Parameter: {}, 'Param': {}, 'Score': {}". format(tune_param, param_config, max_score) )
    return tune_config

def fine_tune_features(clf, ds_vehicles, ds_nonvehicles, n_samples, tune_config, param_grid):
    """
    fine tune the features keeping track of each feature and providing the optimal configuration
    this method is not quite a grid search, as it would be most exaustive
    """
    tune_dict = {}
    params = ['color_space', 'pix_per_cell', 'cells_per_block', 'orient', 'hog_channel',
              'spatial_size', 'hist_bins']
    for k in params:
        f_tune = feature_tune(clf, ds_vehicles, ds_nonvehicles, n_samples, tune_config, param_grid, param=k)
        df_tune = pd.DataFrame.from_dict(f_tune)
        tune_config = update_optimal_tune(df_tune, tune_config)
        tune_dict[k] = {'frame':df_tune, 'config':tune_config}
    return tune_dict, tune_config



### Evlaluation per Image
def evaluate_images(files, clf, scalar, overlap=0.5, **params):
    """
    Evaluate the model via sliding windows and classification
    Size of Search Windows be integer multiples of cell size in hog detection
    """
    annot_imgs = []
    xy_window  = (96,96)                # dimensions of window
    xy_overlap = (overlap, overlap)     # amount of overlap
    xs = [200, None]                    # masked min, max values to search for in sliding window
    ys = [400, 656]                     # masked min, max values to search for in sliding window
    for fname in files:
        img = mpimg.imread(fname)
        draw_img = np.copy(img)
        # scale images (png: 0-1, jpg: 0-255): trained on png, reading in jpgs
        img = img.astype(np.float32)/255
        # sliding window operation
        t_start = time.time()
        windows = cvw.slide_window(img, xs, ys, xy_window, xy_overlap)
        # search window space for detection
        hot_windows = cvw.search_windows(img, windows, clf, scalar, **params)
        t_stop = time.time()
        # annotate image
        window_img = viz.draw_boxes(draw_img, hot_windows, color=(0,0,255), thick=6)
        annot_imgs.append(window_img)
        print('Image Search Time: {}, windows: {}'.format(round(t_stop-t_start, 2), len(windows)) )
    return annot_imgs
