import cv2 as cv
import numpy as np
import os

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def find_table_corners(img):
    # Resize the image
    img = rescaleFrame(img, 0.25)
    
    # Pre-processing
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Edge Detection
    edges = cv.Canny(blurred, 50, 150)
    
    # Find Contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Assuming the largest contour in the image is the table
    table_contour = max(contours, key=cv.contourArea)
    
    # Approximate Contour
    epsilon = 0.05 * cv.arcLength(table_contour, True)
    approx = cv.approxPolyDP(table_contour, epsilon, True)
    
    # Draw the detected corners on the original image
    for point in approx:
        cv.circle(img, tuple(point[0]), 10, (0, 255, 0), -1)
        
    return img

if __name__ == '__main__':

    # Directory containing the images
    img_dir = "Computer Vision Assignment/Part_II/tables"

    # Get list of all images in the directory
    img_files = [f for f in os.listdir(img_dir) if f.startswith('Table') and f.endswith('.jpg')]
    img_files.sort()  # Ensure the images are in order

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)

        # Read in an image
        img = cv.imread(img_path)

        # Check if image is loaded correctly
        if img is None:
            print(f"Error: Image {img_file} not loaded!")
            continue

        img = find_table_corners(img)

        # Show the image
        cv.imshow('Table', img)

        # Wait for a key press
        key = cv.waitKey(0)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

    cv.destroyAllWindows()
