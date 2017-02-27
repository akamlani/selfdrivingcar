import numpy as np
import cv2
import pickle
import glob
import os
import argparse

import cv_transf as cvt
import tracker as tr

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

### Operator Functions
def combine_color_threshold(img, **kwargs):
    """
    Combine color thresholds of different isolated chanels
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S,V = (hls[:,:,2], hsv[:,:,2])
    S_binary = cvt.channel_threshold(S, thresh=(120,255))
    V_binary = cvt.channel_threshold(V, thresh=(150,255))
    # for pixels where there is an overlap, set to 1
    color_comb_binary = np.zeros_like(S)
    color_comb_binary[(S_binary == 1) & (V_binary == 1)] = 1
    return color_comb_binary

def combine_gradients(img, **kwargs):
    """
    Combine different methods of gradients: orientation(x,y), magnitude, direction
    """
    grad_x   = cvt.abs_sobel_orient_thresh(img, orient='x', thresh=(12,255))
    grad_y   = cvt.abs_sobel_orient_thresh(img, orient='y', thresh=(25,255))
    grad_mag = cvt.mag_sobel_thresh(img, kernel_size=5, thresh=(25,255))
    grad_dir = cvt.dir_sobel_thresh(img, kernel_size=15, thresh=(0.7,1.3))
    grad_comb = np.zeros_like(grad_dir)
    grad_comb[((grad_x == 1) & (grad_y == 1)) | ((grad_mag == 1) & (grad_dir == 1))] = 1
    return grad_comb



### Pipelines
def pipeline_images(dirname, pattern, ext, method_base=False):
    """
    Iterate through a direcory of images and perform the pipeline sequence
    """
    # Step 1a: load camera matrix and distortion coefficients
    with open('./ckpts/calibration.p', 'rb') as f:
        data_dict = pickle.load(f)
        mtx  = data_dict['mtx']
        dist = data_dict['dist_coef']
        del data_dict

    # create pattern to look for and output tracked dir to store to
    fname = "".join([pattern, '*', ext])
    path = os.path.join(dirname, fname)
    tracked_dir = "_".join([dirname, 'tracked'])
    if not os.path.exists(tracked_dir): os.makedirs(tracked_dir)

    image_names = glob.glob(path)
    fn = pipeline_frame_base if method_base else pipeline_frame
    _ = [fn(mpimg.imread(fname), mtx, dist, fname, tracked_dir) for fname in image_names]


def pipeline_playvideo(fname_input, fname_ouput, **kwargs):
    """
    Play a video clip via processing frame by frame by our pipeline
    Inputs to be processed should be in RGB format

    fname_input:  'project_video.mp4'
    fname_output: 'project_video_tracked.mp4'
    """
    # Step 1a: load camera matrix and distortion coefficients
    with open('./ckpts/calibration.p', 'rb') as f:
        data_dict = pickle.load(f)
        mtx  = data_dict['mtx']
        dist = data_dict['dist_coef']
        del data_dict

    clip = VideoFileClip(fname_input)
    # input to fn should be img
    video_clip = clip.fl_image(lambda x: pipeline_frame(x, mtx, dist))
    video_clip.write_videofile(fname_output, audio=False)


### Utility pipeline functions
def create_fname(fname, dir_path, prefix):
    """
    create a tracked filename given its original name, directory, and prefix
    """
    digits = "".join([x for x in fname if x.isdigit()])
    fname  = "".join([prefix, digits, '.jpg'])
    fname  = os.path.join(dir_path, fname)
    return fname

def save_file(img, prefix, fname=None, tracked_dir=None, save_type=None):
    """
    if given a filename, save with naming convention to the tracked_dir input
    """
    if fname:
        prefix = '_'.join(['tracked', prefix])
        fname = create_fname(fname, tracked_dir, prefix=prefix)
        cv2.imwrite(fname, img) if save_type==None else mpimg.imsave(fname, img)

def pipeline_frame(img, mtx, dist, fname=None, tracked_dir=None):
    """
    Main function to handle an incoming frame to be processed
    """
    warped_img = pipeline_frame_base(img, mtx, dist, fname, tracked_dir)
    # Step 5: determine which pixels are lane line pixels based on conv method
    window_centroids = conv_centers.find_window_centroids(warped_img)
    conv_img = conv_centers.draw_centroids(warped_img, window_centroids)
    save_file(conv_img, 'conv_centroids', fname, tracked_dir)
    # Step 6. annotate radius curvature and offset via window centroids on original img
    left_fitx, right_fitx = curvature.fit_lane_boundaries(window_centroids)
    left_lane, right_lane, inner_lane = curvature.fit_lanes(left_fitx, right_fitx)
    img_lanes = curvature.view_lanes(img, left_lane, right_lane, inner_lane)
    img_annot = curvature.annotate_frame(img_lanes, window_centroids, left_fitx, right_fitx)
    save_file(img_annot, 'curvature', fname, tracked_dir)
    return img_annot

def pipeline_frame_base(img, mtx, dist, fname=None, tracked_dir=None):
    """
    Base pipeline implementation before convolution and curvature methods are applied
    """
    # Step 1b: correct distortions at boundary using serialized calibration
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    save_file(undist_img, 'undistorted', fname, tracked_dir, save_type=-1)
    # Step 2: gradient and color threshold
    gc_binary = pipeline_gradient_color(undist_img)
    save_file(gc_binary, 'gc_binary', fname, tracked_dir)
    # Step 3: Region of Interest Mask
    rows, cols = gc_binary.shape[:2]
    vertices = np.array([[(150, rows),(150,  350),(cols, 350), (cols, rows) ]], dtype=np.int32)
    roi_binary = cvt.region_of_interest(gc_binary, vertices)
    save_file(roi_binary, 'roi_binary', fname, tracked_dir)
    # Step 4: perspective transform to produce warped image
    src, dst, warped_img, M, Minv = pipeline_warped(roi_binary)
    save_file(warped_img, 'warped_binary', fname, tracked_dir)
    return warped_img

def pipeline_gradient_color(img, **kwargs):
    """
    Combine gradient direction andn color channel thresholds
    """
    # retrieve gradient thresholding
    grad_binary = combine_gradients
    # retrieve color thresholding
    c_binary = combine_color_threshold(img)
    p_img    = np.zeros_like(img[:,:, 0])
    p_img[( (grad_binary == 1) | (c_binary == 1) )] = 255
    return p_img



def pipeline_warped(img, **kwargs):
    """
    Create a warped image for a birds eye view
    """
    src, dst = create_perspective_mappings(img)
    # perspective transform to get birds eye view via warped image
    warped, M, Minv = cvt.perspective_transf(img, src, dst)
    return src, dst, warped, M, Minv


def create_perspective_mappings(img):
    """
    create mappings for perspective transform of warped image
    img: should already be an undistorted image
    src: defined points of trapezoid
    dst: should be flat rectangle: approximately size of img
    """
    rows, cols  = img.shape[:2]
    bot_width   = .7
    mid_width   = .1
    height_pct  = .62             #not including full depth
    bottom_trim = .935            #for front of vehicle
    offset      = rows *.25

    src = np.float32([
            [cols*(.5 - mid_width/2),  rows*height_pct],
            [cols*(.55 + mid_width/2), rows*height_pct],
            [cols*(.6 + bot_width/2),  rows*bottom_trim],
            [cols*(.6 - bot_width/2),  rows*bottom_trim] ])
    dst = np.float32([
            [offset, 0],
            [cols - offset, 0],
            [cols - offset, rows],
            [offset, rows] ])
    return src, dst



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Lane Lines Pipeline')
    parser.add_argument('-v', '--video', action='store_true', help='video or test image processing')
    args = parser.parse_args()

    dirname = 'test_images'
    pattern = 'test'
    ext     = '.jpg'
    fnames  = glob.glob(os.path.join(dirname, 'test*.jpg'))

    # define configuration to be used for tracker conv and curvature
    rows, cols = mpimg.imread(fnames[0]).shape[:2]
    params = {  'window_width':     30,
                'window_height':    rows/9,            # rows/n_windows(9)
                'window_margin':    25,
                'ym_per_pix':       10/rows,           # 10m ~ 720 pixels
                'xm_per_pix':       4/800,             # 4m  ~ 800 pixels
                'smooth_factor':    15,
                'img_dim':          (rows,cols) }
    conv_centers = tr.TrackerConv1D(**params)
    curvature    = tr.TrackerCurvature(**params)

    if args.video:
        #process videos based on program inputs
        fname_input    = "videos/project_video.mp4"
        fname_output   = "videos/project_video_tracked.mp4"
        pipeline_playvideo(fname_input, fname_output)

    else:
        # Process against a set of test images
        pipeline_images(dirname, pattern, ext)
