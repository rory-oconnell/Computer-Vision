import cv2
import numpy as np
from itertools import combinations

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

img = cv2.imread('IsolatedTable4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a blank image with the same dimensions as 'img'
blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# Dilate the image
kernel = np.ones((3, 3), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=2)
cv2.imshow("Dilated", gray)

ret,thresh = cv2.threshold(gray,50,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
print("Number of contours detected:", len(contours))

max_area = 0
largest_rectangle_contour = None

# Iterate through the contours to find the largest rectangle
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_rectangle_contour = cnt

# Draw the largest rectangle on the blank image
if largest_rectangle_contour is not None:
    blank_image = cv2.drawContours(blank_image, [largest_rectangle_contour], -1, (0,255,0), 3)

# Show the detected rectangle
cv2.imshow("Largest Rectangle", blank_image)

# # Detect corners
# corners = cv2.goodFeaturesToTrack(cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY), 100, 0.01, 10)
# corners = np.intp(corners)
# 
# # Function to compute the area of a quadrilateral given its vertices
# def compute_area(corner_set):
#     return cv2.contourArea(np.array(corner_set))

# # Find the combination of four corners that form the largest area
# max_area = 0
# best_combination = []
# for comb in combinations(corners, 4):
#     area = compute_area(comb)
#     if area > max_area:
#         max_area = area
#         best_combination = comb
# 
# # Draw the selected corners
# for corner in best_combination:
#     x, y = corner.ravel()
#     cv2.circle(blank_image, (x,y), 10, (0,0,255), -1)

cv2.waitKey(0)
cv2.destroyAllWindows()
