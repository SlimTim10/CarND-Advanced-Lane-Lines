import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def main():
    images = glob.glob('camera_cal/calibration*.jpg')
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    nx = 9                          # Corners in each row
    ny = 6                          # Corners in each column

    mtx, dist = calibrate_camera(images, img_size, nx, ny)

    undistort_images(images, 'output_images/', mtx, dist)

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
        img_size = (img.shape[1], img.shape[0])

        sp = fname.split('\\')
        if len(sp) < 2:
            sp = fname.split('/')
        newname = sp[1][:-4] + '_undist.jpg'

        dst = cv2.undistort(img, mtx, dist, None, mtx)

        cv2.imwrite(output_dir + newname, dst)
        print('Saving ' + output_dir + newname)

if __name__ == '__main__':
    main()
