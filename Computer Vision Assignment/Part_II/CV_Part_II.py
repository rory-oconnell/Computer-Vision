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

        blank = np.zeros(img.shape, dtype='uint8')
        # cv.imshow('Blank', blank)

        # Convert to greyscale
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blur = cv.GaussianBlur(grey, (5,5), cv.BORDER_DEFAULT)
        # cv.imshow('Blur', blur)

        canny = cv.Canny(blur, 125, 175)
        cv.imshow('Canny Edges', canny)

        # ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
        # cv.imshow('Thresh', thresh)

        contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        print(f'{len(contours)} contour(s) found!')

        cv.drawContours(blank, contours, -1, (0,0,255), 1)
        cv.imshow('Contours Drawn', blank)

        cv.imshow('Table', grey)

        # Wait for a key press
        key = cv.waitKey(0)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

    cv.destroyAllWindows()
