import cv2
import numpy as np
from itertools import combinations

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

img = cv2.imread('IsolatedTable5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Dialate the image
kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(img,kernel,iterations = 2)

# Opening and Closing
kernel = np.ones((9, 9), np.uint8)
for i in range(100):
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

max_area = 0
largest_rectangle_contour = None

# Detect corners
corners = cv2.goodFeaturesToTrack(cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY), 100, 0.01, 10)
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

# Show the corners on the Isolated Table image
cv2.imshow("Shapes with Corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Morphology
cv2.imshow("Morphology", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny Edge Detection
edges = cv2.Canny(dilation, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
