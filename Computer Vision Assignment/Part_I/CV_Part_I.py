# Part I.  
# Locate the table tennis ball.  Using colour, regions and shape locate table tennis balls 
# and table tennis tables in the supplied images (See https://www.scss.tcd.ie/Kenneth.Dawson-Howe/Vision/balls.zip for the ball images 
# with ground truth).  You can assume that the ball is either white or orange, may have some printing on it and will be spherical.  
# Ensure that you use techniques which can be used in general (e.g. ideally the techniques would cope with changes in lighting, etc.).  
# Analyse how well your approach works on the static images of the tables provided, and later on the table tennis video.  
# Note that in the report you may need to use some of the Learning and Evaluation section of the course, 
# also in section 9.3 of “Computer Vision with OpenCV”  (when reporting performance).

import cv2 as cv
import numpy as np
import os


def ConvertToGray(img):
    # Converting to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (13, 13), cv.BORDER_DEFAULT)
    return img_gray

def DetectCircles(img):
    # Detecting circles
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=15, maxRadius=80)
    return circles

def Threshold(img):
    # Thresholding
    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    return thresh

def ConvertHSV(img):
    # Converting to HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return img_hsv


if __name__ == '__main__':

    # Directory containing the images
    img_dir = "Computer Vision Assignment\Part_I\Balls"

    # Get list of all images in the directory
    img_files = [f for f in os.listdir(img_dir) if f.startswith('Ball') and f.endswith('.jpg')]
    img_files.sort()  # Ensure the images are in order

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)

        # Read in an image
        img = cv.imread(img_path)

        # Check if image is loaded correctly
        if img is None:
            print(f"Error: Image {img_file} not loaded!")
            continue

        # Converting to grayscale
        img_gray = ConvertToGray(img)

        # Thresholding
        thresh = Threshold(img_gray)

        circles = DetectCircles(img_gray)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

                # Print the center coordinates of the circle and the diameter
                print(i[0], i[1], 2 * i[2])

        cv.imshow('detected circles', img)

        # Wait for a key press
        key = cv.waitKey(0)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

    cv.destroyAllWindows()
