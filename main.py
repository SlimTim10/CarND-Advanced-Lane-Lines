import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from threshold import *

SAVE_UNDISTORT = False
VISUAL_DEBUG = True
PRINT_DEBUG = True

def main():
    images = glob.glob('camera_cal/calibration*.jpg')
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])
    nx = 9                          # Corners in each row
    ny = 6                          # Corners in each column

    mtx, dist = calibrate_camera(images, img_size, nx, ny)

    if SAVE_UNDISTORT == True:
        undistort_images(images, 'output_images/', mtx, dist)

    images = glob.glob('test_images/*.jpg')
    for fname in images:
        img = cv2.imread(fname)

        img = cv2.undistort(img, mtx, dist, None, mtx)

        img = threshold_image(img)

        img = birds_eye_crop(img)

        if VISUAL_DEBUG == True:
            cv2.imshow('img', img)
            cv2.waitKey(0)

        find_lanes(img)

def calibrate_camera(chessboard_images, img_size, corners_row=9, corners_col=6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((corners_row*corners_col, 3), np.float32)
    objp[:,:2] = np.mgrid[0:corners_row,0:corners_col].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for fname in chessboard_images:
        if PRINT_DEBUG == True:
            print('Reading ' + fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (corners_row,corners_col), None)

        # If found, add object points, image points
        if ret == True:
            if PRINT_DEBUG == True:
                print('Found {} chessboard corners'.format(len(corners)))
            objpoints.append(objp)
            imgpoints.append(corners)

            if VISUAL_DEBUG == True:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (corners_row,corners_col), corners, ret)

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
        if PRINT_DEBUG == True:
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

def birds_eye_crop(img):
    img_size = (img.shape[1], img.shape[0])

    # Perspective source
    src = np.float32([
        [550,470],
        [770,470],
        [1100,660],
        [220, 660]])
    # Offset from borders
    offset = 10
    # Perspective destination
    dst = np.float32([
        [offset, offset],
        [img_size[0]-offset, offset],
        [img_size[0]-offset, img_size[1]-offset],
        [offset, img_size[1]-offset]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

# Return left and right fit lines for found lanes
def find_lanes(binary_warped):
    # Take a histogram of bottom half
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # For drawing visualizations
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find peak of left and right halves to determine left and right lane lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if VISUAL_DEBUG == True:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Visualizations
    if VISUAL_DEBUG == True:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        cv2.imshow('img', out_img)
        cv2.waitKey(0)
        plt.imshow(out_img, cmap='gray')
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_fit, right_fit

if __name__ == '__main__':
    main()
