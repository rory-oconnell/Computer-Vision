import cv2
import numpy as np

# Load the image
image = cv2.imread('Ball6.jpg')

# Convert the image to HLS
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

# Extract the L channel
h, l, s = cv2.split(hls)

# Adjust the luminance values. Here, we increase the luminance by adding a value.
# Make sure not to go beyond the 8-bit range [0, 255].
l = np.clip(l + 100, 0, 255)  # increase luminance by 50; adjust as needed

# Merge the channels back
hls_adjusted = cv2.merge((h, l, s))

# Convert back to BGR
image_adjusted = cv2.cvtColor(hls_adjusted, cv2.COLOR_HLS2BGR)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Adjusted Luminance', image_adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()
