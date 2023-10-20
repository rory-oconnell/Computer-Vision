import cv2
import numpy as np

# Load the image
image = cv2.imread('Ball10.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use adaptive thresholding to create a binary mask of the brightest areas
# Note: You can adjust the values here based on the specifics of your image
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) # Threshold value 230 is an example; adjust accordingly.

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# If no contours are detected, exit
if not contours:
    print("No bright regions found!")
    exit()

# Sort contours by area in descending order and take the largest one
c = max(contours, key=cv2.contourArea)

# Create an all-black mask
mask = np.zeros_like(gray)

# Fill the detected contour in the mask
cv2.drawContours(mask, [c], -1, (255), thickness=cv2.FILLED)

# Bitwise the original image and the mask to get the isolated brightest object
result = cv2.bitwise_and(image, image, mask=mask)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Isolated Brightest Object', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
