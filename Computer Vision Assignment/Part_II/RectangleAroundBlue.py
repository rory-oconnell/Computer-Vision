import cv2
import numpy as np
from itertools import combinations

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

img = cv2.imread('IsolatedTable3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Opening and Closing
kernel = np.ones((9, 9), np.uint8)
for i in range(100):
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


# Create a blank image with the same dimensions as 'img'
blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

max_area = 0
largest_rectangle_contour = None

# Detect corners
corners = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 0.01, 10)
corners = np.intp(corners)

# Function to compute the area of a quadrilateral given its vertices
def compute_area(corner_set):
    return cv2.contourArea(np.array(corner_set))# Find the combination of four corners that form the largest area
max_area = 0
best_combination = []
for comb in combinations(corners, 4):
    area = compute_area(comb)
    if area > max_area:
        max_area = area
        best_combination = comb

# Draw the selected corners
for corner in best_combination:
    x, y = corner.ravel()
    cv2.circle(img, (x,y), 3, (0,0,255), -1)

cv2.imshow("Shapes with Corners", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
