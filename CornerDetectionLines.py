import cv2
import numpy as np
from itertools import combinations

def intersection_point(rho, theta, x_max, y_max):
    a = np.cos(theta)
    b = np.sin(theta)
    x0, y0 = a * rho, b * rho

    # Intersection with top border (y=0)
    if b != 0:
        x_top = (rho - (0 * b)) / a
    else:  # Avoid division by zero
        x_top = np.inf

    # Intersection with bottom border (y=y_max)
    if b != 0:
        x_bottom = (rho - (y_max * b)) / a
    else:  # Avoid division by zero
        x_bottom = np.inf

    # Intersection with left border (x=0)
    if a != 0:
        y_left = (rho - (0 * a)) / b
    else:  # Avoid division by zero
        y_left = np.inf

    # Intersection with right border (x=x_max)
    if a != 0:
        y_right = (rho - (x_max * a)) / b
    else:  # Avoid division by zero
        y_right = np.inf

    points = []
    if 0 <= x_top <= x_max:
        points.append((int(x_top), 0))
    if 0 <= x_bottom <= x_max:
        points.append((int(x_bottom), y_max))
    if 0 <= y_left <= y_max:
        points.append((0, int(y_left)))
    if 0 <= y_right <= y_max:
        points.append((x_max, int(y_right)))

    return points

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

# Closing
kernel = np.ones((9, 9), np.uint8)
for i in range(100):
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

# Hough Line Transform
edges = cv2.Canny(dilation, 100, 200)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# Extend the lines across the entire image
for line in lines:
    rho, theta = line[0]
    intersections = intersection_point(rho, theta, img.shape[1], img.shape[0])
    if len(intersections) == 2:  # We need exactly two points to draw a line
        cv2.line(img, intersections[0], intersections[1], (0, 0, 255), 2)

# Draw the lines on the Isolated Table image
cv2.imshow("Shapes with Lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Morphology
cv2.imshow("Morphology", dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()