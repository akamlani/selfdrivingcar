import numpy as np
import cv2

from scipy.ndimage.measurements import label
import viz_utils as viz


class Vehicle(object):
    def __init__(self):
        self.detected = False           # was vehicle detected in last iteration
        self.n_detections = 0           # number of times this vehicle has been seen
        self.n_nondetections = 0        # number of consecutive times this hard has not been detected
        self.xpixels = None             # pixel x values of last detection
        self.ypixels = None             # pixel y values of last detection
        self.recent_xfitted = []        # x position of last n fits of the bounding box
        self.recent_yfitted = []        # y position of last n fits of bounding box
        self.recent_wfitted = []        # width position of last n fits of bounding box
        self.recent_hfitted = []        # height position of last n fits of bounding box
        self.bestx = None               # average x position of last n fits
        self.besty = None               # average y position of last n fits
        self.bestw = None               # average width of last n fits
        self.besth = None               # average height of last n fits


class F2FTracker(object):
    def __init__(self, dimensions, window_size=10):
        """
        window_size: 1 for single image, else window over multiple frames
        """
        self.nframes = 0                    # frame_cnt
        self.window_size = window_size      # nframes
        self.threshold = 0 if window_size == 1 else 1
        rows, cols = dimensions
        self.heatmap = np.zeros((rows, cols, window_size), dtype=np.float32)

    def process_frame(self, base_img, heatmap_coords):
        # get current heatmap
        window_idx = self.nframes % self.window_size
        heat_curr = viz.add_heat(base_img, heatmap_coords)
        self.heatmap[:, :, window_idx] = heat_curr
        # create a smooth heatmap over a window of frames
        curr_slice = self.heatmap[:, :, :self.nframes + 1]
        item =  curr_slice if self.nframes < self.window_size else self.heatmap
        heat_smooth = np.mean(item, axis=2)
        # improve heatmap instances
        heat_thresh = viz.apply_threshold(heat_smooth, threshold=1)
        # annotate image via heatmap
        labels = label(heat_thresh)
        draw_img = viz.draw_labeled_bboxes(base_img, labels)
        self.nframes += 1
        return draw_img, heat_thresh, labels
