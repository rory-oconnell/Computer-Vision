import cv2
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations

# Define the angle similarity threshold
ANGLE_THRESHOLD = np.deg2rad(10)  # 10 degrees tolerance

def are_angles_similar(theta1, theta2, threshold=ANGLE_THRESHOLD):
    # Normalize angles to be within 0 to pi
    theta1 = np.mod(theta1, np.pi)
    theta2 = np.mod(theta2, np.pi)
    
    # Calculate absolute difference in angles
    angle_diff = np.abs(theta1 - theta2)
    
    # Check if angles are within the threshold, considering angle wraparound
    return angle_diff <= threshold or np.abs(angle_diff - np.pi) <= threshold

# Calculate the area of a polygon given its vertices
def polygon_area(corners):
    n = len(corners) # Number of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# Function to find the four corners of the largest quadrilateral
def find_largest_quadrilateral(intersection_points):
    max_area = 0
    max_quad = None
    # Compute all possible quadrilaterals
    for quad in combinations(intersection_points, 4):
        # Calculate the area of the quadrilateral
        area = polygon_area(quad)
        # Check if this quadrilateral has the largest area so far
        if area > max_area:
            max_area = area
            max_quad = quad
    return max_quad

# Function to calculate intersection of two lines given in Hough transform form.
def line_intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    return int(np.round(x0)), int(np.round(y0))

# Function to check if a point is inside the image boundaries.
def is_point_inside_image(x, y, img_shape):
    return x >= 0 and y >= 0 and x < img_shape[1] and y < img_shape[0]

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

original_image = cv2.imread("Computer Vision Assignment/Part_II/Tables/Table3.jpg")
hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

roi = cv2.imread("Computer Vision Assignment/Part_II/Tables/Table3_BPJ.png")
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Histogram ROI 
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)

# Filtering remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.filter2D(mask, -1, kernel)
_, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

mask = cv2.merge((mask, mask, mask))
result = cv2.bitwise_and(original_image, mask)
cv2.imshow("Result", rescaleFrame(result, 0.25))
cv2.imshow("Mask", rescaleFrame(mask, 0.25))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Dialate the image
kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(result,kernel,iterations = 2)

# Opening and Closing
kernel = np.ones((9, 9), np.uint8)
for i in range(100):
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

# Closing
kernel = np.ones((9, 9), np.uint8)
for i in range(100):
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Mask", rescaleFrame(dilation, 0.25))
cv2.waitKey(0)
cv2.destroyAllWindows()


# Hough Line Transform
edges = cv2.Canny(dilation, 100, 200)
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

# Extend the lines across the entire image
for line in lines:
    rho, theta = line[0]
    intersections = intersection_point(rho, theta, result.shape[1], result.shape[0])
    if len(intersections) == 2:  # We need exactly two points to draw a line
        cv2.line(result, intersections[0], intersections[1], (0, 0, 255), 2)

# List to hold the coordinates of the intersection points
intersection_points = []

for line1, line2 in combinations(lines, 2):
    try:
        # Only calculate intersection if the angles of the lines are not too similar
        if not are_angles_similar(line1[0][1], line2[0][1]):
            x, y = line_intersection(line1, line2)
            if is_point_inside_image(x, y, original_image.shape):
                intersection_points.append((x, y))
    except np.linalg.LinAlgError:
        # Lines are parallel and cannot be intersected
        continue

# Mark the intersection points on the image
for point in intersection_points:
    cv2.circle(result, point, radius=5, color=(0, 255, 0), thickness=-1)  # -1 thickness fills the circle

# Assuming intersection_points is a list of tuples (x, y) of intersection points
largest_quad = find_largest_quadrilateral(intersection_points)

# Check if a largest quad was found and print or draw it
if largest_quad:
    # Convert points to float32
    src_pts = np.array(largest_quad, dtype="float32")

    # Define the destination points for the "doubled" dimensions
    # The destination image will be twice as large as the standard table size in pixels
    # Standard table size is 274 cm x 152.5 cm, this is approximated to 400 x 600 pixels
    dst_pts = np.array([[0, 0],
                        [750, 0],
                        [750, 750],
                        [0, 750]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # The size of the output image will be 400 x 600 pixels
    warped_image = cv2.warpPerspective(original_image, M, (800, 800))

    cv2.imshow("Plan View", warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No quadrilateral found.")