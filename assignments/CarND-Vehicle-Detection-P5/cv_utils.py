import numpy as np
import cv2



def read_image(image_filename):
    """
    Read in an image in RGB format, OpenCV uses BGR format
    """
    return mpimg.imread(image_filename)


def scale_image(img, scale=1.0):
    """
    Spatially scale an image rather than a specific given size
    """
    rows, cols = img.shape[:2]
    img_scaled = cv2.resize(img, (np.int(cols/scale), np.int(rows/scale)) )
    return img_scaled

def resize_image(img, size=(32, 32), log=False):
    """
    Spatially downsample an image to a given size
    """
    small_img = cv2.resize(img, size)
    #small_img = sc.misc.resize(image, size)
    if log: print("Original Dim:{}, New Dimensions:{}".format(img.shape, small_img.shape))
    return small_img


def featurevec_flatten(img):
    """
    Create a 1D Feature Vector from an given img
    """
    feature_vec = img.ravel()
    print(feature_vec.shape)
    return feature_vec


#### OpenCV references
# http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
# http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html
