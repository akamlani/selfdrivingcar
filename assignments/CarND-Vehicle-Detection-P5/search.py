import numpy as np
import cv2
import cv_features as cvf
import cv_utils as cvu
import viz_utils as viz


class Search(object):
    def __init__(self, scaler, **config):
        ### for window search
        self.update_config(**config)
        self.scaler = scaler

    def update_config(self, **config):
        """
        update configuration parameters
        """
        self.color_space  = config['color_space']
        self.spatial_size = config['spatial_size']
        self.hist_bins = config['hist_bins']
        self.ys = config['ys']
        self.xs = config['xs']
        self.scale = config['scale']
        self.orient = config['orient']
        self.pix_per_cell = config['pix_per_cell']
        self.cells_per_block = config['cells_per_block']
        self.cells_per_step  = config['cells_per_step']
        self.window = config['window']


    def config_hog_search(self, img):
        """
        Configuration to perform a hog search to find positive samples in a given image
        """
        img = img.astype(np.float32)/255                                    # scale image: (png: 0-1, jpg: 0-255)
        img_crop = img[self.ys[0]:self.ys[1], :, :]                         # crop img to relative dimensions (ROI)
        clr_trf  = cvf.transform_colorspace(img_crop, self.color_space)     # transform color space from RGB
        clr_trf  = cvu.scale_image(clr_trf, self.scale)                     # scale rather than selecting diffferent window sizes
        ch1, ch2, ch3 = [clr_trf[:,:,ch] for ch in range(3)]                # HOG operates on per channel basis

        # get hog features as multi-dimensional array per channel for the entire image
        params={'orient':self.orient, 'pix_per_cell':self.pix_per_cell, 'cells_per_block':self.cells_per_block}
        params.update({'vis':False, 'feature_vec':False})
        hog1,hog2,hog3 = [cvf.get_hog_features(ch, **params) for ch in ([ch1,ch2,ch3])]
        # block configurations
        self.nfeat_per_block    = self.orient*self.cells_per_block**2
        self.nblocks_per_window = (self.window //self.pix_per_cell) -1
        # number_blocks: define number of hog cells across an image according to channel dimensions
        # number_steps:  define number of steps to occur across hog array to extract features
        rows, cols = ch1.shape[:2]
        number_blocks = lambda max_dim, pix_per_cell: (max_dim // pix_per_cell) -1
        number_steps  = lambda nblocks, nblocks_window, cells_step: (nblocks - nblocks_window)//cells_step
        self.nxblocks, self.nyblocks = number_blocks(cols, self.pix_per_cell), number_blocks(rows, self.pix_per_cell)
        nxsteps, nysteps  = \
        [number_steps(it, self.nblocks_per_window, self.cells_per_step) for it in (self.nxblocks, self.nyblocks) ]
        return clr_trf, (nxsteps, nysteps), (hog1,hog2,hog3)

    def subsample_hog_features(self, img, clf, nsteps, hog_img_features):
        """
        Perform Search based on extracted Image Hog Features across a series of patches
        """
        bbox_coords = []; heatmap_coords = []
        nxsteps, nysteps = nsteps
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*self.cells_per_step
                xpos = xb*self.cells_per_step
                # extract image patch via defining in image space
                xleft, ytop = (xpos*self.pix_per_cell, ypos*self.pix_per_cell)
                clr_trf_patch = img[ytop:ytop+self.window, xleft:xleft+self.window]
                patch_img = cvu.resize_image(clr_trf_patch, size=(64,64))
                # Get color features for a given patch
                spatial_features = cvf.bin_spatial(patch_img, size=self.spatial_size)
                hist_features    = cvf.color_histogram(patch_img,  nbins=self.hist_bins)
                # Get HOG Features for given patch: patches have discontinuties at edges: different gradient
                hog_features = [ hog_it[ypos:ypos+self.nblocks_per_window, xpos:xpos+self.nblocks_per_window].ravel()
                                 for hog_it in (hog_img_features) ]
                hog_features = np.hstack((hog_features))
                # concatenate features and perform classifier prediction
                patch_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1)
                patch_features_sc = self.scaler.transform(patch_features)
                # perform prediction, check for positive instance detection
                #yhat = clf.predict(patch_features_sc)
                yhat = clf.predict_proba(patch_features_sc)
                pred_pos = yhat[:, 1] > 0.95
                if pred_pos:
                    bbox_coord, heatmap_coord = self.create_coords(xleft, ytop)
                    bbox_coords.append((bbox_coord))
                    heatmap_coords.append((heatmap_coord))

        return bbox_coords, heatmap_coords



    def create_coords(self, xleft, ytop):
        """
        Create Bounding Box and Heatmap Coordinates for a positive instance detection
        """
        xbox_left = np.int(xleft*self.scale)
        ytop_draw = np.int(ytop*self.scale)
        win_draw  = np.int(self.window*self.scale)
        ystart    = self.ys[0]

        startx, starty = xbox_left, np.int(ytop_draw+ystart)
        endx, endy = np.int(xbox_left+win_draw), np.int(ytop_draw+win_draw+ystart)
        bbox_coord = ((startx, starty), (endx, endy))
        #bbox_coords.append(((startx, starty), (endx, endy)))

        startx, starty = ytop_draw+ystart, ytop_draw+win_draw+ystart
        endx, endy = xbox_left, xbox_left+win_draw
        heatmap_coord = ((startx, starty), (endx, endy))
        #heatmap_coords.append(((startx, starty), (endx, endy)))
        return bbox_coord, heatmap_coord
