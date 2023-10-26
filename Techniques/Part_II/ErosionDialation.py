# Part II. 
# Locate the table tennis table.  You must now locate the corners of the table (the outside of the white lines) 
# using edge detection, and then transform the image so that you have a plan view of the table.  
# Determine how well your approach works on the static images of the tables provided 
# (See https://www.scss.tcd.ie/Kenneth.Dawson-Howe/Vision/tables.zip for the table images with ground truth).  
# Ensure that you use techniques which can be used in general (e.g. ideally the techniques would cope with changes in lighting, etc.).  
# Analyse how well your approach works on the static images of the tables provided, and later on the table tennis video.  
# Note that in the report you may need to use some of the Learning and Evaluation section of the course, 
# also in section 9.3 of “Computer Vision with OpenCV”  (when reporting performance).

import cv2 as cv
import numpy as np
import os

def rescaleFrame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

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

        # Resize the image
        img = rescaleFrame(img, 0.25)

        # Greyscale the image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for i in range(5):
            # Erode the image
            kernel = np.ones((5, 5), np.uint8)
            img = cv.erode(img, kernel, iterations=1)

            # Dilate the image
            kernel = np.ones((5, 5), np.uint8)
            img = cv.dilate(img, kernel, iterations=1)

        # Use canny edge detection
        edges = cv.Canny(img,50,150,apertureSize=3)
        
        # Apply HoughLinesP method to 
        # to directly obtain line end points
        lines_list =[]
        lines = cv.HoughLinesP(
                    edges, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=100, # Min number of votes for valid line
                    minLineLength=5, # Min allowed length of line
                    maxLineGap=10 # Max allowed gap between line for joining them
                    )
        
        # Iterate over points
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1,y1),(x2,y2)])

        # Show the image
        cv.imshow('Table', img)

        # Wait for a key press
        key = cv.waitKey(0)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

    cv.destroyAllWindows()
