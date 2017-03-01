import numpy as np
import cv2

### Sobel Operations
def sobel_operator(img, orient, kernel_size=3, **kwargs):
    # smooths as weights middle more, outer parts less
    gs_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # derivative x dir: edge detector in vertical
    if orient == 'x':
        grad = cv2.Sobel(gs_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    # derivative x dir: edge detector in vertical
    if orient == 'y':
        grad = cv2.Sobel(gs_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # scaled to 8-bit as np.unit8
    if kwargs['scale']:
        abs_grad = np.absolute(grad)
        grad = np.uint8(255*abs_grad/np.max(abs_grad))
    return grad

def abs_sobel_orient_thresh(img, orient, thresh, kernel_size=3):
    thresh_min, thresh_max = thresh
    abs_grad = sobel_operator(img, orient, kernel_size, **{'scale': True})
    binary_img = np.zeros_like(abs_grad)
    binary_img[(abs_grad >= thresh_min) & (abs_grad <= thresh_max)] = 1
    return binary_img

def dir_sobel_thresh(img, kernel_size, thresh=(0, np.pi/2)):
    thresh_min, thresh_max = thresh
    grad_x = sobel_operator(img, orient='x', kernel_size=kernel_size, **{'scale': False})
    grad_y = sobel_operator(img, orient='y', kernel_size=kernel_size, **{'scale': False})
    abs_grad_dir = np.arctan2(np.absolute(grad_y), np.absolute(grad_x))
    binary_img = np.zeros_like(abs_grad_dir)
    binary_img[(abs_grad_dir >= thresh_min) & (abs_grad_dir <= thresh_max)] = 1
    return binary_img

def mag_sobel_thresh(img, kernel_size, thresh=(0,255)):
    grad_x = sobel_operator(img, orient='x', kernel_size=kernel_size, **{'scale': False})
    grad_y = sobel_operator(img, orient='y', kernel_size=kernel_size, **{'scale': False})
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # Rescale to 8 bit
    scale_factor = np.max(grad_mag)/255
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    thresh_min, thresh_max = thresh
    binary_img = np.zeros_like(grad_mag)
    binary_img[(grad_mag >= thresh_min) & (grad_mag <= thresh_max)] = 1
    return binary_img

### Utility Masking
def region_of_interest(img, vertices):
    # assumes single channel
    mask    = np.zeros_like(img)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

### Color Thresholds
def color_threshold(img, thresh=(90,255)):
    thresh_min, thresh_max = thresh
    binary_img = np.zeros_like(channel_img)
    binary_img[(channel_img >= thres_min) & (channel_img <= thresh_max)] = 1
    return binary_img

def channel_threshold(channel, thresh):
    # R Channel: thresh=(200,255); R=rgb[:,:,0]
    # H Channel: thresh=(15,100),  H=hsv[:,:,0]
    # S Channel: thresh=(90,255);  S=hls[:,:,2] ***
    thresh_min, thresh_max = thresh
    binary_img = np.zeros_like(channel)
    binary_img[(channel >= thresh_min) & (channel <= thresh_max)] = 1
    return binary_img

### Perspective Transforms
def perspective_transf(img, src, dst, **kwargs):
    """
    z = distance from camera
    perspective transform: transforms the z coordinate to view image from different perspective/angle
    background: (large z: further away), foreground: (small z, closer to camera)
    birds-eye view: zoom in on farther objects
    """
    rows, cols = img.shape[:2]
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    Minv = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (cols, rows), flags=cv2.INTER_LINEAR)
    opt = True if 'image_only' in kwargs and kwargs['image_only'] else False
    return warped if opt else (warped, M, Minv)
