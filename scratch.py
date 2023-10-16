import cv2 as cv
import numpy as np
import os

def Threshold(img):
    # Thresholding
    ret, thresh = cv.threshold(img, 150, 255, cv.THRESH_BINARY)
    return thresh

def GreyScale(img):
    # Converting to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (3,3), cv.BORDER_DEFAULT)
    return img_gray

def DetectCircles(img):
    # Detecting circles
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=15, maxRadius=80)
    return circles

def ConvertHSV(img):
    # Converting to HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return img_hsv

img = cv.imread("Ball1.jpg")
cv.imshow('img', img)

# Convert to HSV
img_hsv = ConvertHSV(img)
cv.imshow('hsv', img_hsv)

# Define two hsv colour values to threshold
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# grey = GreyScale(img)
# cv.imshow('grey', grey)s
# 
# thresh = Threshold(grey)
# cv.imshow('thresh', thresh)
# 
# circles = DetectCircles(thresh)
# 
# # Draw the detected circles
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0,:]:
#         # draw the outer circle
#         cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         # draw the center of the circle
#         cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
#     cv.imshow('detected circles', img)

cv.waitKey(0)