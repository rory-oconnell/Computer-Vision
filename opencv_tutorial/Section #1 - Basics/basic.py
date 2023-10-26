import cv2 as cv
import os

img_path = 'opencv-course/Resources/Photos/park.jpg'
if not os.path.exists(img_path):
    raise Exception(f"Image {img_path} does not exist!")
img = cv.imread(img_path)

# print("Current working directory:", os.getcwd())
#
# img = cv.imread('Resources\Photos\park.jpg')
# cv.imshow('Park', img)
#
# # Converting to grayscale
# grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow("Grey", grey)
#
# # Blur - Kernel size has to be an odd number
# blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
# cv.imshow('Blur99', blur)
#
# # Edge Cascade
# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges', canny)
#
# # Dilating the image
# dilated = cv.dilate(canny, (3,3), iterations=1)
# cv.imshow('Dilated', dilated)
#
# # Eroding
# eroded = cv.erode(dilated, (3,3), iterations=1)
# cv.imshow('Eroded', eroded)
#
# # Resize
# resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized', resized)
#
# # Cropping
# cropped = img[50:200, 200:400]
# cv.imshow('Cropped', cropped)
#
# cv.waitKey(0)