import cv2
import numpy as np

def find_table_corners(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce image noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges in the image
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours based on area in descending order and keep the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    
    # Approximate the contour to a polygon and check if it has 4 points (for a rectangle)
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            for corner in approx:
                cv2.circle(img, tuple(corner[0]), 10, (0, 0, 255), -1)  # Draw a red circle on each corner

            # Display the result
            cv2.imshow("Detected Corners", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return approx.reshape(4, 2)
    return None

# Test the function
corners = find_table_corners("Techniques\Part_II\Table4.jpg")

if corners is not None:
    print("Corners found:")
    for corner in corners:
        print(corner)
else:
    print("Could not detect table corners.")
