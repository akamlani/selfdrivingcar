import numpy as np
import glob
import cv2
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def camera_chessboard(dirname, pattern, ext, nx=9, ny=6):
    """
    draw chessboard corners on original image based on (nx,ny) mapping from 2D to 3D
    2D = image points, 3D=object points
    """
    fnames     = []                                     # tuple of file name, corner file name
    img_points = []                                     # 2D points in image plane
    obj_points = []                                     # 3D points in real world
    obj_p = np.zeros((ny*nx,3), np.float32)             # 3 columns, z-axis intiialized to zeros (board on flat image plane)
    obj_p[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)  # x,y coordinates of corners, reshape into 2D

    fname = "".join([pattern, '*', ext])
    path = os.path.join(dirname, fname)
    corners_dir = "_".join([dirname, 'corners'])
    if not os.path.exists(corners_dir): os.makedirs(corners_dir)

    images = glob.glob(path)
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        gs_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # search for corners in grayscale image
        ret, corners = cv2.findChessboardCorners(gs_img, (nx,ny), None)
        if ret == True:
            obj_points.append(obj_p)
            img_points.append(corners)
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            # filename digits match source file
            digits = "".join([x for x in fname if x.isdigit()])
            fname_corners  = "".join(["cal_corners", digits, ext])
            fname_corners  = os.path.join(corners_dir, fname_corners)
            fnames.append((fname, fname_corners))
            mpimg.imsave(fname_corners, img)
    return img_points, obj_points, fnames

def camera_calibrate(img_points, obj_points, img_size, **kwargs):
    """
    3D (object points) to 2D (image points): changes shape and size of 3D objects
    http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    """
    # dist=distortion coeff, mtx=camera matrix, position=(rvecs: rotation, tvecs: translational)
    ret, mtx, dist_coef, rvecs, tvecs = \
    cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    return mtx, dist_coef



if __name__ == "__main__":
    # get camera corners
    img_points, obj_points, fnames = \
    camera_chessboard('camera_cal', 'calibration', '.jpg', nx=9, ny=6)
    # calibrate camera
    img = mpimg.imread(fnames[0][0])
    img_size = (img.shape[::-1])[:2]
    mtx, dist_coef = camera_calibrate(img_points, obj_points, img_size)
    # serialize camera matrix(mtx), distortion coefficients to disk
    if not os.path.exists('./ckpts'): os.makedirs('./ckpts')
    with open('./ckpts/calibration.p', 'wb') as f:
        pickle.dump({'mtx': mtx, 'dist_coef': dist_coef}, f)
