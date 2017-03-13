import numpy as np
import cv2
import matplotlib.image as mpimg
import cv_features as cvf


### sliding window
def slide_window(img, xs, ys, window_size=(64,64), window_overlap=(0.5,0.5)):
    """
    Perform sliding window method
    """
    window_list = []
    # get region of interest dimensions
    get_min = lambda x, idx: 0 if x[idx] == None else x[idx]
    get_max = lambda img, x, idx, shape_idx: img.shape[shape_idx] if x[idx] == None else x[idx]
    xs = [get_min(xs, 0), get_max(img, xs, 1, 1)]
    ys = [get_min(ys, 0), get_max(img, ys, 1, 0)]
    # Compute the span of the region to be searched
    x_span, y_span = (xs[1] - xs[0], ys[1] - ys[0])
    # compute number of pixels per step in x,y
    x_window,  y_window  = window_size
    x_overlap, y_overlap = ( 1-window_overlap[0], 1-window_overlap[1] )
    nx_pixperstep, ny_pixperstep = ( np.int(x_window*x_overlap), np.int(y_window*y_overlap) )
    nx_windows, ny_windows = (np.int(x_span/nx_pixperstep)-1, np.int(y_span/ny_pixperstep)-1 )
    # iterate through image, finding x,y window positions (consider windows one-by-one w/classifier)
    for ys_idx in range(ny_windows):
        for xs_idx in range(nx_windows):
            # Calculate window position
            xs_start = int( xs_idx*nx_pixperstep + xs[0] )
            xs_end   = xs_start + x_window
            ys_start = int( ys_idx*ny_pixperstep + ys[0] )
            ys_end   = ys_start + y_window
            window_list.append(((xs_start, ys_start), (xs_end, ys_end)))
    return window_list

### Search window space
def search_windows(img, windows, clf, scaler, **params):
    """
    Search windows for positive detection based on classifier
    Method searches per an image, not per aregion
    """
    pos_windows = []
    for window in windows:
        # resize given image based on training data dimensions
        size = (64,64)
        sample_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], size)
        # extract features
        features = cvf.fetch_features(sample_img, **params)
        # scale features
        scaled_features = scaler.transform(np.array(features).reshape(1,-1))
        # prediction via classifier
        prediction = clf.predict(scaled_features)
        if prediction == 1: pos_windows.append(window)

    return pos_windows
