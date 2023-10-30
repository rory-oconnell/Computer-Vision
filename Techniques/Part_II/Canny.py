import cv2
import numpy as np

def rescaleFrame(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Read the image
image = cv2.imread("IsolatedTable.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarize the image (you may need to adjust the threshold values)
_, thresholded = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

# Use Canny edge detection
edges = cv2.Canny(thresholded, 50, 150)

# Optionally, dilate the lines to make them more prominent
dilated = cv2.dilate(edges, None, iterations=1)

# Display the result
cv2.imshow("Isolated Lines", rescaleFrame(dilated))
cv2.waitKey(0)
cv2.destroyAllWindows()
