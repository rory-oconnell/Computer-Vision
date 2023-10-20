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

        cv.imshow('Table', img)

        # Wait for a key press
        key = cv.waitKey(0)

        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

    cv.destroyAllWindows()
