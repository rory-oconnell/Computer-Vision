import cv2
import numpy as np
from matplotlib import pyplot as plt

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

original_image = cv2.imread("Computer Vision Assignment\Part_II\Tables\Table1.jpg")
hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

roi = cv2.imread("Computer Vision Assignment\Part_II\BlueTable.png")
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

hue, saturation, value = cv2.split(hsv_roi)

# Histogram ROI
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)

# Filtering remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_and(original_image, mask)

cv2.imshow("Mask", rescaleFrame(mask, 0.25))
cv2.imshow("Original image", rescaleFrame(original_image, 0.25))
cv2.imshow("Result", rescaleFrame(result, 0.25))
cv2.imshow("Roi", rescaleFrame(roi, 0.25))
cv2.waitKey(0)
cv2.destroyAllWindows()