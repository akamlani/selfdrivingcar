import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg


degree_to_radian = lambda angle: angle * np.pi / 180
read_grayscale   = lambda fname: cv2.imread(fname, 0)

def thresh_channel(channel, thresholds):
    binary_im = np.zeros_like(channel)
    thresh_im = binary_im[(channel > thresholds[0]) & (channel <= thresholds[1])] = 1
    return thresh_im

def binarize_image(im, **kwargs):
    # mask and threshold an image
    # OPT 1: via greyscale
    if kwargs['opt'] == 'gray':
        thresh = (180,255)
        gs_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        thresh_im = thresh_channel(gs_im, thresh)
    # Opt 2: via RGB channels
    elif kwargs['opt'] == 'RGB_ch_R':
        thresh = (200, 255)
        R,G,B = [im[:,:,ch] for ch in range(3)]
        thresh_im = thresh_channel(R, thresh)
    # Opt 3: via HLS channels
    elif kwargs['opt'] == 'HLS_ch_S':
        # more robust to lighting conditions
        thresh = (90, 255)
        hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
        H,L,S = [hls[:,:,ch] for ch in range(3)]
        thresh_im = thresh_channel(S, thresh)
    elif kwargs['opt'] == 'HLS_ch_H':
        thresh = (15, 100)
        hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
        H,L,S = [hls[:,:,ch] for ch in range(3)]
        thresh_im = thresh_channel(H, thresh)

    return thresh_im

def resize_image(im, **kwargs):
    # cv2.INTER_AREA: shrink; cv2.INTER_LINEAR: zoom; cv2.INTER_CUBIC:  slow
    # Default: cv2.INTER_LINEAR
    height, width = im.shape[:2]
    k = kwargs['scale']
    return cv2.resize(im,(k*width, k*height), interpolation = cv2.INTER_LINEAR)

def resize_images(features, size=(64,64)):
    return map(lambda x: cv2.resize(x.squeeze(), size), features)


def flip_image(im, y, **kwargs):
    # ALT: np.fliplr(image)
    # horizontal flipping about y-axis: avoid model being biased towards particular direction
    axis_flip = 1 if kwargs['axis']=='vertical' else 0
    # transform image, on all images (not just center image)
    return (cv2.flip(im, axis_flip), y*-1.0 if axis_flip == 1 else y)

def rotate_image(im, y, **kwargs):
    # alernatively flip image vertically as appropriate: np.flipud(data)
    rows,cols = im.shape[:2]
    center = (cols/2,rows/2)
    angle   = np.random.uniform(low=kwargs['min'], high=kwargs['max'])
    radian  = degree_to_radian(angle)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_im  = cv2.warpAffine(im, rot_mat, (cols,rows), flags=cv2.INTER_NEAREST)
    return (rot_im, y - radian)

def shift_image(im, y, **kwargs):
    # x_min=-5, x_max=5
    # apply shift in horizontal direction
    rows,cols  = im.shape[:2]
    # horizontal shift: shift between [-0.2,0.2]
    shift_x = np.random.uniform(-kwargs['width_range'],  kwargs['width_range'])
    tr_x    = np.ceil(shift_x*cols).astype(np.int64)
    # vertical shift: simulate slope
    shift_y = np.random.uniform(-kwargs['height_range'], kwargs['height_range'])
    tr_y    = np.ceil(shift_y*rows).astype(np.int64)
    tr_m = np.float32([[1,0,tr_x],[0,1,tr_y]])
    # avoid crop of image by adding amount shifted, however we need to resize image
    tr_im = cv2.warpAffine(im, tr_m, (cols,rows))
    return (tr_im, y - shift_x)

def brightness_image(im, **kwargs):
    # adjust brightness via HSV V channel
    # adj_val = = 0.25; [0.93,1.4]
    adj_val = np.random.uniform(kwargs['min'], kwargs['max'])
    # adjust V channel for lightness/brightness via addition
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    random_bright = adj_val+ np.random.uniform()
    hls[:,:,1]    = hls[:,:,1]*random_bright
    adj_im = cv2.cvtColor(hls,cv2.COLOR_HLS2RGB)
    return adj_im

def hist_equalize_image(im, **kwargs):
    # improves contrast of image via stretching histogram to ends: apply when confined to particular region
    # won't work well where when there are both bright and dark pixels present
    # equalize the Y channel
    yuv_im = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    yuv_im[:,:,0] = cv2.equalizeHist(yuv_im[:,:,0])
    return cv2.cvtColor(yuv_im, cv2.COLOR_YUV2RGB)

def gaussian_blur_image(im, **kwargs):
    kernel_size = np.random.choice([3,5,7])
    return cv2.GaussianBlur(im,(kernel_size,kernel_size),0)

def speckle_image(im, **kwargs):
    rows,cols = im.shape[:2]
    # add some noise to the image, alternatively, use a gaussian gaussian
    noise = np.random.randint(kwargs['min'],kwargs['max'],(rows, cols))
    jitter = np.zeros_like(im)
    jitter[:,:,1] = noise
    tr_im = cv2.add(im, jitter)
    return tr_im

def gamma_corr_image(im, **kwargs):
    # let gamma = 0.25-1.15 as input
    # brighten image: gamma > 1, darken image: gamma < 1
    gamma_corr = np.random.uniform(kwargs['min'], kwargs['max'])
    inv_gamma = 1.0/gamma_corr
    table = np.array( [ ((idx / 255.0) ** inv_gamma) * 255
    		            for idx in np.arange(0, 256)] ).astype(np.uint8)
    return cv2.LUT(im, table)


def tr_color_image(im, **kwargs):
    """
        # YUV: Separate intensity from color: Y=luminance/instensity (Grayscale), U/V=color info
        # HSV: Hue,Saturation,Value: separate most prioparty properties of colors
        color_space = {
            cv2.COLOR_RGB2HSV,
            cv2.COLOR_RGB2LUV,
            cv2.COLOR_RGB2HLS,
            cv2.COLOR_RGB2YUV,
            cv2.COLOR_RGB2GRAY,
            # opencv cv2.imread assumes input in BGR format
            cv2.COLOR_BGR2YUV,
            cv2.COLOR_BGR2HSV,
            cv2.COLOR_BGR2RGB,
            cv2.COLOR_BGR2GRAY
        }
    """
    # pprint( [x for x in dir(cv2) if x.startswith('COLOR_')] )
    return cv2.cvtColor(im, kwargs['color_space'])

### Interface function
def augment_image(img_file, img_target, fn, **kwargs):
    # AlT: Keras Generator: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    img_data   = mpimg.imread(img_file)
    # perform actual function transformation
    fn_str_name  = "_".join(fn.__name__.split('_')[:-1])
    aug_str_name = "_".join([fn_str_name, 'augmented.'])
    if  'adjust_target' in kwargs and kwargs['adjust_target']:
        img_fn_data, img_fn_target = fn(img_data, img_target, **kwargs)
    else:
        img_fn_target = img_target
        img_fn_data   = fn(img_data, **kwargs)

    # save image with '{fname}_{fn}_augmented.{ext}'
    ext = 'jpg'
    fname = img_file.strip().split('.' +ext)[0]
    fname_aug  = "_".join([fname, aug_str_name]) + ext
    mpimg.imsave(fname_aug, img_fn_data)
    return ( fname_aug, img_fn_target)
