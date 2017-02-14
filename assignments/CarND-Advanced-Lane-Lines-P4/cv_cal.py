import numpy as np
import glob
import cv2
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def camera_calibrate(img_points, obj_points, img_size, **kwargs):
    # http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # 3D (object points) to 2D (image points): changes shape and size of 3D objects

    # dist=distortion coeff, mtx=camera matrix, position=(rvecs: rotation, tvecs: translational)
    ret, mtx, dist_coef, rvecs, tvecs = \
    cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    dst_img = cv2.undistort(img, mtx, dist_coef, None, mtx)
    return mtx, dist_coef, dst_img



if __name__ == "__main__":
    obj_points = [] # 3D points in real world
    img_points = [] # 2D points in image plane

    nx,ny = (9,6)
    obj_p = np.zeros((ny*nx,3), np.float32)
    obj_p[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    images = glob.glob('./camera_cal/calibration*.jpg')
    if not os.path.exists('./camera_cal_corners'): os.makedirs('./camera_cal_corners')
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        gs_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # search for corners in grayscale image
        ret, corners = cv2.findChessboardCorners(gs_img, (nx,ny), None)
        if ret == True:
            obj_points.append(obj_p)
            img_points.append(corners)
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            # fielname digits match source file
            digits = "".join([x for x in fname if x.isdigit()])
            fname  = "".join(["cal_corners", digits, ".jpg"])
            fname = os.path.join('camera_cal_corners', fname)
            mpimg.imsave(fname, img)

    # calibrate camera
    img = mpimg.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    mtx, dist_coef, undist_img = camera_calibrate(img_points, obj_points, img_size)
    # serialize camera matrix(mtx), distortion coefficients to disk
    if not os.path.exists('./ckpts'): os.makedirs('./ckpts')
    with open('./ckpts/calibration.p', 'wb') as f:
        pickle.dump({'mtx': mtx, 'dist_coef': dist_coef}, f)
