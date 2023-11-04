import cv2
import numpy as np

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Load the image
image = cv2.imread("Techniques\Part_II\Table5.jpg")

# Convert the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for blue color in HSV
# These values may need tweaking depending on the exact shade of blue of your table
lower_blue = np.array([75, 40, 40])
upper_blue = np.array([155, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image to isolate the blue table
res = cv2.bitwise_and(image, image, mask=mask)

# Dilate the image
kernel = np.ones((5, 5), np.uint8)
res = cv2.dilate(res, kernel, iterations=1)

# Rescale the images for display
image = rescaleFrame(image, 0.25)
res = rescaleFrame(res, 0.25)
                   
# Display the original and result images
cv2.imshow('Original Image', image)
cv2.imshow('Isolated Table', res)
cv2.imwrite('IsolatedTable5.jpg', res)

# Save the result
cv2.waitKey(0)
cv2.destroyAllWindows()