import numpy as np
import cv2
import pipeline as pl


def find_hist_peaks(warped_img):
    """
    Find most prominant L,R peaks of histogram as starting point for lane line curves
    Starting base pts to perform sliding window placed around line centers
    Each peak is considered a 'bump' to start
    """
    rows, cols = warped_img.shape
    # add up pixels of each column in lower shap of image
    histogram   = np.sum(warped_img[int(rows/2):,:], axis=0)
    midpoint    = np.int(histogram.shape[0]/2)
    leftx_base  = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    print('Peaks(x) left: {}, midpoint: {}, right: {}'.format(leftx_base, midpoint, rightx_base))
    return histogram, leftx_base, rightx_base


class TrackerConv1D(object):
    """
    Class to determine which pixels below to a given line via pixel position
    Performs a sliding window search operation via 1D convolution to find best window center positions
    Convolution method maxmizes number of 'hot' pixels in each window

    Future Actions: TBD: Comparison w/curvature, disregard those w/bad curvature
    """
    def __init__(self, window_width, window_height, window_margin,
                 ym_per_pix=1, xm_per_pix=1, smooth_factor=15, img_dim=(None,None)):
        self.recent_centers = []            # store past (l,r) center set values used for smoothing/avg output

        # count pixels inside center values to determine curve values
        self.window_width  = window_width           # window pixel width of center values
        self.window_height = window_height          # window pixel height of center values
        self.margin = window_margin                 # pixel distance in both directions to slide (l,r)
        self.ym_per_pix = ym_per_pix                # meters per pixel in (y) vertical axis
        self.xm_per_pix = xm_per_pix                # meters per pixel in (x) horizontal axis
        self.smooth_factor = smooth_factor
        self.img_ratio = 0.75                       # define part of image window to extract
        self.min_pix   = 50                         # define minimum number of pixels to move from last position

    def search_window_space(self, signal, line_center, cols):
        """
        search within window using prior convolutional signal line to extract new line center
        convolution signal reference is at right side of window, not center of window
        """
        # margin is the amount in pixels the window is allowed to slide around
        # only search within window (line center) rather an entire conv space
        # as on the right half of conv signal, max value is downwards sloped (hence min value)
        center  = self.window_width/2
        min_idx = int(max(line_center + center - self.margin, 0))
        max_idx = int(max(line_center + center + self.margin, cols))
        line_center = np.argmax(signal[min_idx:max_idx]) + min_idx - center
        return line_center


    def window_mask(self, img_ref, center, level):
        """
        create an image mask for visualization
        """
        rows, cols = img_ref.shape[:2]
        window_width, window_height, window_center = (self.window_width, self.window_height, self.window_width/2)
        output = np.zeros_like(img_ref)
        output[ int(rows - (level+1)*window_height):int(rows - level*window_height),
                max(0, int(center - window_center)):min(int(center + window_center),cols) ] = 1
        return output

    def find_window_centroids(self, warped_img):
        """
        Warped Image: Input image is already color/gradient thresholded image and perspective transformed
        Finding l,r centroid points via convolution: peak is highest overlap of pixels, most likely position
        Sliding window template across Image (L to R), overlappig values summed together

        Divide image into (n_windows=9) vertical slides based on height
        Number of centroids is based on number of windows(9) according to img dimensions height
        """
        rows, cols    = warped_img.shape[:2]
        img_x_center  = int(cols/2)
        window_height, window_center = (self.window_height, self.window_width/2)
        window_centroids = []                   # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width)     # Create our window template that we will use for convolutions

        # convolve the vertical image slice with the window template, maximum value w/max overlap
        # sum quarter bottom of image (hence image_ratio~0.75) to get slice
        l_sum    = np.sum(warped_img[int(self.img_ratio*rows):, :img_x_center], axis=0)
        r_sum    = np.sum(warped_img[int(self.img_ratio*rows):, img_x_center:], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum)) - window_center
        r_center = np.argmax(np.convolve(window,r_sum)) - window_center + img_x_center
        window_centroids.append((l_center,r_center))

        # Get Subsequent layers for max pixel locations
        n_windows = int(rows/window_height)
        for level in range(1, n_windows):
            # convolve the window into the vertical slice of the image, vertical slice moves upwards
            image_layer = np.sum(warped_img[int(rows-(level+1)*window_height):int(rows-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # using prior history, select best centroid for each line
            l_center = self.search_window_space(conv_signal, l_center, cols)
            r_center = self.search_window_space(conv_signal, r_center, cols)
            # centroids in terms of x position, y position defined in height of convolution window
            window_centroids.append((l_center,r_center))
        self.recent_centers.append(window_centroids)
        # moving average of last n frames(smooth_factor): avoids the estimated poisition from being noisy and fluctuating
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)

    def draw_centroids(self, warped_img, window_centroids):
        """
        Using give window centroids, overlay them on a given warped image
        """
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped_img)
        r_points = np.zeros_like(warped_img)
        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = self.window_mask(warped_img, window_centroids[level][0], level)
    	    r_mask = self.window_mask(warped_img, window_centroids[level][1], level)
    	    # Add graphic points from window mask here to total pixels found
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8)                 # add both left and right window pixels together
        zero_ch  = np.zeros_like(template)                              # create a zero color channel
        color_ch = cv2.merge((zero_ch,template,zero_ch))                # window pixels green via template in channel of RGB format
        template = np.array(color_ch,np.uint8)
        color_ch = cv2.merge((warped_img,warped_img,warped_img))        # making original road pixels 3 color channels
        warpage  = np.array(color_ch,np.uint8)
        result   = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)      # overlay orignal road image with window results
        return result



class TrackerCurvature(object):
    """
    Track Lane Boundaries and Cuvature from given windows centroids via Conv 1D Method
    """
    def __init__(self, window_width, window_height, window_margin,
                 ym_per_pix=1, xm_per_pix=1, smooth_factor=15,
                 img_dim=(None,None)):
        self.window_width  = window_width           # window pixel width of center values
        self.window_height = window_height          # window pixel height of center values
        self.ym_per_pix = ym_per_pix                # meters per pixel in (y) vertical axis
        self.xm_per_pix = xm_per_pix                # meters per pixel in (x) horizontal axis
        self.rows, self.cols = img_dim


        # repsresent vertical size, resolution to ~1px (continuous space)
        self.yvals      = np.linspace(0, self.rows, num=self.rows)
        # fit to centers from top to bottom (hence reverse)
        self.res_yvals  = np.arange(self.rows-(window_height/2),0,-window_height)


    def view_lanes(self, img, left_lane, right_lane, inner_lane):
        """
        project the lane lines (left,right) on the original frame
        """
        rows, cols = img.shape[:2]
        road = np.zeros_like(img)
        road_bkg = np.zeros_like(img)
        cv2.fillPoly(road, [left_lane],  color=[255,0,0])              # set left lane as red
        cv2.fillPoly(road, [right_lane], color=[0,0,255])              # set left lane as blue
        #cv2.fillPoly(road, [inner_lane], color=[0,255,0])              # set inner lane as green
        cv2.fillPoly(road_bkg, [left_lane],  color=[255,255,255])
        cv2.fillPoly(road_bkg, [right_lane], color=[255,255,255])
        # warp road images and background
        src, dst, warped, M, Minv = pl.pipeline_warped(img)
        road_warped = cv2.warpPerspective(road, Minv, (self.cols, self.rows), flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, (self.cols, self.rows), flags=cv2.INTER_LINEAR)
        # show weighted versions
        base   = cv2.addWeighted(img,  1.0, road_warped_bkg, -1.0, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
        return result


    def fit_lanes(self, left_fitx, right_fitx):
        """
        Provide line lanes a depth for visualization
        """
        # set markers and overlay on original image, img is original image
        set_lanes = lambda x_line, y_pix, window_width: np.array(
            list( zip( np.concatenate((x_line-self.window_width/2, x_line[::-1]+self.window_width/2), axis=0),
                       np.concatenate((y_pix, y_pix[::-1]), axis=0) )) , np.int32 )

        left_lane  = set_lanes(left_fitx,  self.yvals, self.window_width)
        right_lane = set_lanes(right_fitx, self.yvals, self.window_width)
        inner_lane = set_lanes(right_fitx, self.yvals, self.window_width)
        return left_lane, right_lane, inner_lane


    def fit_lane_boundaries(self, window_centroids):
        """
        fit the lane boundaries to the left,right window centroids positions via 2nd order polynomial
        create a continuous value for both left and right fit based on height of image
        """
        window_height = self.window_height
        rows, cols    = (self.rows, self.cols)
        # fit a polynomial for curvature
        leftx,rightx = self.get_xlane(window_centroids)
        # fit to box centers working its way up the image
        left_fitx  = self.fit_poly(leftx,  self.yvals, self.res_yvals)   # fit polynomial to left line center
        right_fitx = self.fit_poly(rightx, self.yvals, self.res_yvals)   # fit polynomail to right line center
        return left_fitx, right_fitx

    def fit_curvature(self, x_line):
        """
        calculate radius of curvature:
        if line is straight: large curvature, else small curvature
        R = 1/K(dtheta/dx)
        R = (1 + (2*A*y + B)**2)**1.5 /abs(2*A)
        http://www.intmatch.com/applications-differentiation/8-radius-curvature.php
        http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC
        """
        # fit 2nd order poly curvature in terms of meters, not pixels
        x_line_m = np.array(x_line, np.float32)*self.xm_per_pix
        y_val_m  = np.array(self.res_yvals, np.float32)*self.ym_per_pix
        curve_fit_cr = np.polyfit(y_val_m, x_line_m, 2)
        # calculate curvature
        A, B = (curve_fit_cr[0], curve_fit_cr[1])
        ym   = self.yvals[-1]*self.ym_per_pix
        curverad = ((1 + (2*A*ym + B)**2)**1.5) / np.absolute(2*A)
        return curverad

    def fit_poly(self, x_line, yvals, res_yvals):
        """
        fit a 2nd degree polynomial for a continuous value(f(y) not f(x))
        Lines in curvature near vertical: may have same x for more than one y, calculate f(y)
        f(y) = A*y**2 + B*y + C
        """
        line_fit  = np.polyfit(res_yvals, x_line, 2)
        line_fitx = line_fit[0]*(yvals*yvals) + line_fit[1]*yvals + line_fit[2]
        line_fitx = np.array(line_fitx, np.int32)
        return line_fitx

    def get_xlane(self, window_centroids):
        """
        utility function to extract left and right lane centroids from a given window centroids
        number of window centroids based on height of a window
        """
        centroid_lr   = lambda wc, level: (wc[level][0], wc[level][1])
        leftx, rightx = zip(*[ centroid_lr(window_centroids, level) for level in range(0, len(window_centroids)) ] )
        return leftx, rightx

    def find_center_offset(self, left_fitx, right_fitx):
        """
        find offset from lane center: distance from center of lane at bottom of image
        lane center is assumed to be camera center
        """
        # calculate offset of the car on the road
        camera_center = (left_fitx[-1] + right_fitx[-1])/2            # -1 index: position closes to car, average: for center
        center_diff   = (camera_center-self.cols/2)*self.xm_per_pix   # scale offset from being offcenter by xm
        side_pos      = 'right' if center_diff <= 0 else 'left'       # left or right dependent on positive or negative
        return center_diff, side_pos

    def annotate_frame(self, img, window_centroids, left_fitx, right_fitx):
        """
        pre-condition: should be an already overladyed image via function view_lanes

        Annotate the original image with center offset and radius curvature
        Radius curvature based on window centroids for a given line
        """
        img = np.copy(img)
        # find the vehicle offset from assumed center camera
        center_diff, lane = self.find_center_offset(left_fitx, right_fitx)
        # calculate radius of curvature
        leftx, rightx = self.get_xlane(window_centroids)
        left_radius_curv   = self.fit_curvature(leftx)
        right_radius_curv  = self.fit_curvature(rightx)
        radius_curv = (left_radius_curv + right_radius_curv) / 2
        # place additional annotations for plot
        place_text = lambda img, fmt, pos: cv2.putText(img, fmt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        fmt_curve  = 'Radius of Curvature = {}m'.format(round(radius_curv,3))
        fmt_offset = 'Vehicle is: {}m {} of center'.format(abs(round(center_diff,3)), lane)
        place_text(img, fmt_curve,  (50,50))
        place_text(img, fmt_offset, (50,100))
        return img
