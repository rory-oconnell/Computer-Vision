import cv2
import numpy as np

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def find_table_corners(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Resize the image
    img = rescaleFrame(img, 0.25)
    
    # Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the largest contour in the image is the table
    table_contour = max(contours, key=cv2.contourArea)
    
    # Approximate Contour
    epsilon = 0.05 * cv2.arcLength(table_contour, True)
    approx = cv2.approxPolyDP(table_contour, epsilon, True)
    
    # Draw the detected corners on the original image
    for point in approx:
        cv2.circle(img, tuple(point[0]), 10, (0, 255, 0), -1)
        
    cv2.imshow('Detected Table Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

find_table_corners('Computer Vision Assignment/Part_II/tables/Table1.jpg')