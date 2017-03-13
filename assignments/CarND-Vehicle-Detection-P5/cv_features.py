import numpy as np
import cv2

from skimage.feature import hog
from skimage import color, exposure

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import viz_utils as viz

### Features
def get_hog_features(img, orient, pix_per_cell, cells_per_block, vis=False, feature_vec=True):
    """
    Histogram of Oriented Gradients: histogram of distinct gradient directions/orientations within the cell
    Signature as 1D Array: robust to variations in shape, can be used for signature of any shape
    Each pixel gets a vote into which histogram bin it belongs in

    http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog
    http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
    """
    return hog(img,                                                 #Grayscale or Single Channel Image
               orientations=orient,                                 #(6-9) number of orientation bins gradient split up into
               pixels_per_cell=(pix_per_cell, pix_per_cell),        #cell size over which gradient histogram is computed
               cells_per_block=(cells_per_block, cells_per_block),  #local area which histogram counts per cell normalized
               transform_sqrt=False,                                #reduce effects of shadows or illumination variation(error w/neg)
               visualise=vis,                                       #produces a visualization image
               feature_vector=feature_vec                           #automatically unrolls features in output
    )

def bin_spatial(img, size=(32,32)):
    """
    Downsample Image of each channel: Reduce Spaital dimensions and flatten
    Assumes 3D Image
    """
    return np.hstack( tuple([cv2.resize(img[:,:,ch], size).ravel() for ch in range(3)]) )


def color_histogram(img, nbins=32, **kwargs):
    """
    Compute histogram of color channels separately
    Variations in size can be accomodated via normalization of histogram
    """
    # Take histograms fore each channel (returns counts, edges=bin intervals)
    ch1_hist, ch2_hist, ch3_hist = \
    [np.histogram(img[:,:,channel], bins=nbins) for channel in range(3)]
    # concentrate features
    # channel index: ch{i}_hist[0]: counts per bin, ch{i}_hist[1]: bin edges
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))
    if 'bin_centers' in kwargs and kwargs['bin_centers']:
        # calculate bin center regions: not using due to png/jpg format differences
        bin_centers = (ch1_hist[1][1:] + ch1_hist[1][0:len(ch1_hist[1])-1])/2
        return ch1_hist, ch2_hist, ch3_hist, bin_centers, hist_features
    else:
        return hist_features

def transform_colorspace(img, color_space='RGB2YCrCb'):
    """
    Convert the color space
    """
    if color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return feature_image



### Feature Pipelines
def fetch_features(img, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                   orient=9, pix_per_cell=8, cells_per_block=2, hog_channel=0, vis=False):
    """
    Core implmementation of feature extraction
    Ordering applied, classifier w/same order of extraction of features
    For single sample testing operations on single window
    """
    feature_img = transform_colorspace(img, color_space) if color_space != 'RGB' else np.copy(img)
    # Extract spatial color features
    spatial_features = bin_spatial(feature_img, size=spatial_size)
    # Extract Features for Color Space
    color_features = color_histogram(feature_img, nbins=hist_bins)
    # HOG features, specific channel or handle all channels of a feature image
    params={'orient':orient, 'pix_per_cell':pix_per_cell, 'cells_per_block':cells_per_block}
    params.update({'vis':vis, 'feature_vec':True})
    if hog_channel == 'ALL':
        n_channels   = feature_img.shape[2]
        hog_features = [get_hog_features(feature_img[:,:,ch], **params) for ch in range(n_channels)]
        # convert to 1D feature vector (via ravel operation)
        hog_features = np.ravel(hog_features)
    else:
        if vis: hog_features, hog_img = get_hog_features(feature_img[:,:,hog_channel], **params)
        else: hog_features = get_hog_features(feature_img[:,:,hog_channel], **params)
        hog_features = np.ravel(hog_features)
    # To be used for creating a feature list
    features = np.concatenate((spatial_features, color_features, hog_features))
    if vis: return features, hog_img
    else: return features


def extract_features(filenames, color_space='RGB', spatial_size=(32, 32), hist_bins=32,
                     orient=9, pix_per_cell=8, cells_per_block=2, hog_channel=0, vis=False):
    """
    Pipeline to perform feature extraction from an list of files
    For batch training related operations
    """
    # Convert image to new color space (if specified, e.g. 'HSV', 'LUV')
    features_l = []
    for filename in filenames:
        img = mpimg.imread(filename)
        features = fetch_features(img, color_space, spatial_size, hist_bins,
                                  orient, pix_per_cell, cells_per_block, hog_channel, vis)
        # Append the new feature vector to the features list
        features_l.append(features)
    return features_l



### Visualizations
fetch_img = lambda ds: mpimg.imread(ds[np.random.randint(0, len(ds))])
def show_hog_features(ds, axi1, axi2, title, **params):
    """
    visualize the hog image features
    """
    base_img = fetch_img(ds)
    features, hog_img = fetch_features(base_img, **params)
    axi1.imshow(base_img)
    viz.set_axi_opts(axi1, **{'title': '{} type'.format(title)} )
    axi2.imshow(hog_img, cmap='hot')
    viz.set_axi_opts(axi2, **{'title': 'HOG, {} type'.format(title)} )
    return base_img, hog_img, features

def show_color_hist(img, color_space, nbins=32, scale=1.0):
    """
    visualize color histogram distributions per channel
    """
    feature_img = transform_colorspace(img, color_space) if color_space != 'RGB' else np.copy(img)
    ch1_hist, ch2_hist, ch3_hist, bin_centers, hist_features = \
    color_histogram(feature_img, nbins, **{'bin_centers': True})
    # plot channels
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16,3))
    channels = [ch1_hist, ch2_hist, ch3_hist]
    titles   = [color_space[it] for it in range(len(color_space))]
    for axi, (title, chi_hist) in zip( (ax1,ax2,ax3), zip(titles, channels)):
        axi.bar(bin_centers*scale, chi_hist[0]*scale)
        axi.set_xlim(0, scale)
        axi.set_title("Channel: {}".format(title))
    plt.tight_layout()


def show_spatial_bin(img, axi, color_space, size=(32,32)):
    """
    visualize spatial dimenionality reduction as a flattened 1D Feature Vector
    """
    feature_img = transform_colorspace(img, color_space) if color_space != 'RGB' else np.copy(img)
    spatial_features = bin_spatial(feature_img, size=size)
    axi.plot(spatial_features)
    axi.set_title("Spatial Size: {}".format(size))

def show_color_space(axi, img, color_space='RGB', sample_size=64):
    """
    visualize subsampled pixels across a 3D plot, channel per axis
    """
    rows, cols = img.shape[:2]
    # subsample small fraction of pixels to visualize
    scale = max(rows, cols, sample_size) / sample_size  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(cols/scale), np.int(rows/scale)), interpolation=cv2.INTER_NEAREST)
    # scale to [0,1] colors for plotting
    img_small_colors = img_small
    # transform color space
    img_small_tr = transform_colorspace(img, color_space) if color_space != 'RGB' else np.copy(img)
    viz.show_3d(axi, img_small_tr, img_small_colors, color_space)
