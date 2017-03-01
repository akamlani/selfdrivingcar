import numpy as np
import cv2
import pickle
import glob
import os
import argparse

import cv_transf as cvt
import lines as lf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from moviepy.editor import VideoFileClip

### Operator Functions
def combine_color_threshold(img, **kwargs):
    """
    Combine color thresholds of different isolated chanels
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    S,V,L,B = (hls[:,:,2], hsv[:,:,2], luv[:,:,0], lab[:,:,2])
    L_binary = cvt.channel_threshold(L, thresh=(125,255))  # detect white  lines
    B_binary = cvt.channel_threshold(B, thresh=(145,200))  # detect yellow lines
    S_binary = cvt.channel_threshold(S, thresh=(110,255))  # 100
    V_binary = cvt.channel_threshold(V, thresh=(130,255))   # 50
    # for pixels where there is an overlap, set to 1
    color_comb_binary = np.zeros_like(S)
    # color_comb_binary[( (S_binary == 1) & (V_binary == 1) ) ] = 1
    color_comb_binary[( (S_binary == 1) & (V_binary == 1) ) |
                      ( (L_binary == 1) & (B_binary == 1) ) ] = 1
    return color_comb_binary

def fetch_gradients(img, **kwargs):
    """
    Combine different methods of gradients: orientation(x,y), magnitude, direction
    """
    grad_x   = cvt.abs_sobel_orient_thresh(img, orient='x', thresh=(12,255))
    grad_y   = cvt.abs_sobel_orient_thresh(img, orient='y', thresh=(25,255))
    grad_mag = cvt.mag_sobel_thresh(img, kernel_size=3,     thresh=(25,255))
    grad_dir = cvt.dir_sobel_thresh(img, kernel_size=15,    thresh=(0.7,1.3))
    return grad_x, grad_y, grad_mag, grad_dir


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
    undist_img, warped_img, M, Minv = pipeline_frame_base(img, mtx, dist, fname, tracked_dir)
    # Step 5: determine which pixels are lane line pixels based on conv method
    window_centroids = conv_centers.find_window_centroids(warped_img)
    conv_img = conv_centers.draw_centroids(warped_img, window_centroids)
    save_file(conv_img, 'conv_centroids', fname, tracked_dir)
    # Step 6. annotate radius curvature and offset via window centroids on original img
    left_fitx, right_fitx = curvature.fit_lane_boundaries(window_centroids)
    left_lane, right_lane, inner_lane = curvature.fit_lanes(left_fitx, right_fitx)
    img_lanes = curvature.view_lanes(undist_img, Minv, left_lane, right_lane, inner_lane)
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
    roi_binary = pipeline_roi(gc_binary)
    save_file(roi_binary, 'roi_binary', fname, tracked_dir)
    # Step 4: perspective transform to produce warped image
    src, dst, warped_img, M, Minv = pipeline_warped(roi_binary)
    save_file(warped_img, 'warped_binary', fname, tracked_dir)
    return undist_img, warped_img, M, Minv

def pipeline_gradient_color(img, **kwargs):
    """
    Combine gradient direction andn color channel thresholds
    """
    # retrieve gradient thresholding
    grad_x, grad_y, grad_mag, grad_dir = fetch_gradients(img)
    #grad_cond = ((grad_x == 1) & (grad_y == 1))
    #grad_cond = ( ((grad_x == 1) & (grad_y == 1)) | (grad_mag == 1) )
    grad_cond = ( ((grad_x == 1) & (grad_y == 1)) | ((grad_mag == 1) & (grad_dir == 1)) )
    # retrieve color thresholding
    c_binary = combine_color_threshold(img)
    p_img    = np.zeros_like(img[:,:, 0])
    p_img[( (grad_cond) | (c_binary == 1) )] = 255
    return p_img

def pipeline_roi(img, **kwargs):
    """
    Perform a Region of Interest Mask on particular region
    """
    rows, cols = img.shape[:2]
    vertices = np.array([[(500, 400),(100,rows),(cols,rows),(900, 400)]], dtype=np.int32)
    roi_binary = cvt.region_of_interest(img, vertices)
    return roi_binary

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
    bottom_trim = 1.0
    # src = np.float32([[570, 460],[235, int(720*bottom_trim)], [1145, int(720*bottom_trim)],[735, 460]])
    # dst = np.float32([[320, 0], [320,  720], [960, 720],[960, 0]])

    src = np.float32([[575, 460],[185, int(720*bottom_trim)], [1200, int(720*bottom_trim)],[740, 460]])
    dst = np.float32([[320, 0], [320,  720], [960, 720],[960, 0]])
    return src, dst



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Lane Lines Pipeline')
    parser.add_argument('-v', '--video', action='store_true', help='video processing')
    args = parser.parse_args()

    dirname = 'test_images'
    pattern = 'test'
    ext     = '.jpg'
    fnames  = glob.glob(os.path.join(dirname, 'test*.jpg'))

    # define configuration to be used for tracker conv and curvature
    rows, cols = mpimg.imread(fnames[0]).shape[:2]
    params = {  'window_width':     25,                #25
                'window_height':    rows/9,            # rows/n_windows(9)
                'window_margin':    25,                #25
                'ym_per_pix':       10/rows,           # 10m ~ 720 pixels
                'xm_per_pix':       4/600,             # 4m  ~ 600 pixels
                'smooth_factor':    15,
                'img_dim':          (rows,cols) }
    conv_centers = lf.LineConv1D(**params)
    curvature    = lf.LineCurvature(**params)

    if args.video:
        #process videos based on program inputs
        fname_input    = 'videos/project_video.mp4'
        fname, ext     = fname_input.split('.')
        fname_output   = "_".join([fname, 'tracked']) + "".join(['.', ext])
        print('Video Input: {}, Output: {}'.format(fname_input, fname_output))
        pipeline_playvideo(fname_input, fname_output)
    else:
        # Process against a set of test images
        pipeline_images(dirname, pattern, ext)
