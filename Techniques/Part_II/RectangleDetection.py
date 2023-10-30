import cv2
import numpy as np

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

img = cv2.imread('IsolatedTable.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a blank image with the same dimensions as 'img'
blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# Dilate the image
kernel = np.ones((5, 5), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=10)

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

# Draw the largest rectangle and label it on the blank image
if largest_rectangle_contour is not None:
    x, y, w, h = cv2.boundingRect(largest_rectangle_contour)
    ratio = float(w)/h
    if ratio >= 0.9 and ratio <= 1.1:
        blank_image = cv2.drawContours(blank_image, [largest_rectangle_contour], -1, (0,255,255), 3)
        cv2.putText(blank_image, 'Square', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(blank_image, 'Rectangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        blank_image = cv2.drawContours(blank_image, [largest_rectangle_contour], -1, (0,255,0), 3)

cv2.imshow("Shapes", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
