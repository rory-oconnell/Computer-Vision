# Part II. 
# Locate the table tennis table.  You must now locate the corners of the table (the outside of the white lines) 
# using edge detection, and then transform the image so that you have a plan view of the table.  
# Determine how well your approach works on the static images of the tables provided 
# (See https://www.scss.tcd.ie/Kenneth.Dawson-Howe/Vision/tables.zip for the table images with ground truth).  
# Ensure that you use techniques which can be used in general (e.g. ideally the techniques would cope with changes in lighting, etc.).  
# Analyse how well your approach works on the static images of the tables provided, and later on the table tennis video.  
# Note that in the report you may need to use some of the Learning and Evaluation section of the course, 
# also in section 9.3 of “Computer Vision with OpenCV”  (when reporting performance).

import cv2 as cv
import numpy as np
import os
import math

def HoughLineTransform(img):
    src = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
 
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

            cv.imshow("Source", src)
            cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
            cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

            cv.waitKey()
    return 0

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

if __name__ == '__main__':

    # Directory containing the images
    img_dir = "Computer Vision Assignment/Part_II/tables"

    # Get list of all images in the directory
    img_files = [f for f in os.listdir(img_dir) if f.startswith('Table') and f.endswith('.jpg')]
    img_files.sort()  # Ensure the images are in order

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)

        # Read in an image
        img = cv.imread(img_path)

        # Check if image is loaded correctly
        if img is None:
            print(f"Error: Image {img_file} not loaded!")
            continue

        # Resize the image
        img = rescaleFrame(img, 0.25)

        # Hough Line Transform
        HoughLineTransform(img)

        # Convert to greyscale
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # # Good features to track
        # corners = cv.goodFeaturesToTrack(grey, 50, 0.01, 10) # 4 corners, quality level, min distance between corners

        # # Apply Hough Line Detection
        # lines = cv.HoughLinesP(grey, 1, np.pi/180, 1, minLineLength=100, maxLineGap=100) # The parameters are img, rho, theta, threshold, minLineLength, maxLineGap

        # # Draw the lines
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
        # # Convert corners to integers
        # corners = np.int0(corners)
 
        # # Draw the corners
        # for corner in corners:
        #     x, y = corner.ravel()
        #     cv.circle(img, (x, y), 3, 255, -1)

        cv.imshow('Table', img)

        # Wait for a key press
        key = cv.waitKey(0)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

    cv.destroyAllWindows()
