import cv2
import numpy as np

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# read image
img = cv2.imread("Computer Vision Assignment\Part_II\Tables\Table2.jpg")
h, w, c = img.shape

# get average color of img
color = cv2.mean(img)[0:3]

# convert img to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# do adaptive threshold on gray image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 13)
thresh3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# change white to color
result1 = thresh3.copy()
result1[thresh==255] = color

# optionally colorize text darker and more blue
result2 = result1.copy()
result2[thresh==0] = (color[0],0.65*color[1],0.65*color[2])

# display it
cv2.imshow("IMAGE", rescaleFrame(img, 0.25))
cv2.imshow("THRESHOLD", rescaleFrame(thresh, 0.25))
cv2.imshow("RESULT1", rescaleFrame(result1, 0.25))
cv2.imshow("RESULT2", rescaleFrame(result2, 0.25))
cv2.waitKey(0)