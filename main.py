import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from threshold import *

def main():
    images = glob.glob('camera_cal/calibration*.jpg')
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    nx = 9                          # Corners in each row
    ny = 6                          # Corners in each column

    mtx, dist = calibrate_camera(images, img_size, nx, ny)

    # undistort_images(images, 'output_images/', mtx, dist)

    images = glob.glob('test_images/*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])

        img = cv2.undistort(img, mtx, dist, None, mtx)

        img = threshold_image(img)

        src = np.float32([
            [550,470],
            [770,470],
            [1100,660],
            [220, 660]])
        offset = 10
        dst = np.float32([
            [offset, offset],
            [img_size[0]-offset, offset],
            [img_size[0]-offset, img_size[1]-offset],
            [offset, img_size[1]-offset]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        cv2.imshow('img', warped)
        cv2.waitKey(0)

def calibrate_camera(chessboard_images, img_size, corners_row=9, corners_col=6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((corners_row*corners_col, 3), np.float32)
    objp[:,:2] = np.mgrid[0:corners_row,0:corners_col].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for fname in chessboard_images:
        print('Reading ' + fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (corners_row,corners_col), None)

        # If found, add object points, image points
        if ret == True:
            print('Found {} chessboard corners'.format(len(corners)))
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (corners_row,corners_col), corners, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    return mtx, dist

def undistort_images(images, output_dir, mtx, dist):
    for fname in images:
        img = cv2.imread(fname)

        sp = fname.split('\\')
        if len(sp) < 2:
            sp = fname.split('/')
        newname = sp[1][:-4] + '_undist.jpg'

        dst = cv2.undistort(img, mtx, dist, None, mtx)

        cv2.imwrite(output_dir + newname, dst)
        print('Saving ' + output_dir + newname)

def threshold_image(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    hls_binary = hls_thresh(img, h_thresh=(0, 0), l_thresh=(0, 0), s_thresh=(90, 255))

    ksize = 19

    gradx_binary = abs_sobel_thresh(s_channel, 'x', sobel_kernel=ksize, thresh=(50, 150))
    grady_binary = abs_sobel_thresh(s_channel, 'y', sobel_kernel=ksize, thresh=(10, 255))
    
    mag_binary = mag_thresh(s_channel, sobel_kernel=ksize, thresh=(90, 255))

    dir_binary = dir_threshold(s_channel, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined_binary = np.zeros_like(gradx_binary)
    combined_binary[
        ((gradx_binary == 1) & (grady_binary == 1))
        | ((mag_binary == 1) & (dir_binary == 1))
    ] = 1

    return combined_binary * 255.0


if __name__ == '__main__':
    main()
