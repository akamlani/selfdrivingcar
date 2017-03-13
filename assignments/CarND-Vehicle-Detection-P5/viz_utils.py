import numpy as np
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def set_axi_opts(axi, **kwargs):
    """
    Configure an axis for plotting
    """
    axi.grid(False)
    axi.get_yaxis().set_visible(False)
    axi.get_xaxis().set_ticks([])
    fontsize=12
    if 'fontsize' in kwargs: fontsize=kwargs['fontsize']
    params = {'fontweight':'bold', 'fontstyle':'italic', 'fontsize': fontsize}
    if 'title'  in kwargs: axi.set_title(kwargs['title'], **params)
    if 'xlabel' in kwargs: axi.set_xlabel(kwargs['xlabel'], **params)


def show_3d(ax, pixels, colors, axis_labels):
    """
    Plot 3D distribution of pixels
    """
    scale = 255
    # Set axis limits
    ax.set_xlim(*(0,255/scale))
    ax.set_ylim(*(0,255/scale))
    ax.set_zlim(*(0,255/scale))
    # Set axis labels and sizes
    axis_labels=list(axis_labels)
    #ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=13, labelpad=16)
    ax.set_xticks([0,0.5,1.0])
    ax.set_ylabel(axis_labels[1], fontsize=13, labelpad=16)
    ax.set_yticks([0,0.5,1.0])
    ax.set_zlabel(axis_labels[2], fontsize=13, labelpad=16)
    ax.set_zticks([0,0.5,1.0])
    # Plot pixel values with colors given in colors_rgb
    ax.scatter(pixels[:, :, 0].ravel(),
               pixels[:, :, 1].ravel(),
               pixels[:, :, 2].ravel(),
               c=colors.reshape((-1, 3)), edgecolors='none')
    return ax


def apply_threshold(heatmap, threshold):
    """
    Zero out pixels below a given threshold for the heatmap, for values > 1 < threshold
    This should help with false positives
    """
    heatmap_thresh = np.copy(heatmap)
    ind = np.where(np.logical_and(heatmap_thresh>1, heatmap_thresh<=threshold))
    heatmap_thresh[ind] = 0
    #heatmap_thresh[(heatmap_thresh <= threshold)] = 0
    return heatmap_thresh

def add_heat(img, bbox_list):
    """
    Add heat to a corresponding positive region
    Rejecting areas where there are false positives by appling a threshold
    Should be applied to several frames, not just a single image instance
    Add +=1 for all pixels inside each bounding box
    """
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    for box in bbox_list:
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][0]:box[0][1], box[1][0]:box[1][1]] += 1
        #heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

### Bounding Boxes for each instance
def draw_bboxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Annotate image via drawing bounded boxes on calculated sliding windows
    """
    draw_img = np.copy(img)
    # Draw rectangles given bbox coordinates as opposing coordinates
    # bboxes = opposing coordinates: (x1,y1), (x2,y2)
    [cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick) for bbox in bboxes]
    return draw_img

def draw_labeled_bboxes(img, labels):
    """
    Draw bounding boxes based on given labels
    Better technique that performing blob detection
    Combines all nonzero labels for a single bounding box in a patch region 
    http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    """
    # iterate through all detected instances
    for it in range(1, labels[1]+1):
        # find pixels w/each vehicle label value
        nonzero = (labels[0] == it).nonzero()
        # identify x,y  values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # define a bounding box based on min/max x,y
        bbox = ( (np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)) )
        blue_clr = (0,0,255)
        cv2.rectangle(img, bbox[0], bbox[1], blue_clr, 6)
    return img
