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
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Blur the image
        gray_img = cv.GaussianBlur(gray_img, (5, 5), 0)

        for i in range(5):
            # Erode the image
            kernel = np.ones((5, 5), np.uint8)
            gray_img = cv.erode(gray_img, kernel, iterations=1)

            # Dilate the image
            gray_img = cv.dilate(gray_img, kernel, iterations=1)

        # Use canny edge detection
        edges = cv.Canny(gray_img, 50, 150, apertureSize=3)
        
        # Apply HoughLinesP method
        lines_list = []
        lines = cv.HoughLinesP(
                    edges, 1, np.pi/180, threshold=100, 
                    minLineLength=5, maxLineGap=10)
        
        # Sort the lines based on their length and only consider the top 4
        if lines is not None:
            lines = sorted(lines, key=lambda x: np.sqrt((x[0][0] - x[0][2])**2 + (x[0][1] - x[0][3])**2), reverse=True)
            for points in lines[:10]:
                x1,y1,x2,y2 = points[0]
                cv.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
                lines_list.append([(x1,y1),(x2,y2)])

        # Show the image
        cv.imshow('Table', img)

        # Wait for a key press
        key = cv.waitKey(0)
  
        # If 'q' is pressed, exit the loop
        if key == ord('q'):
            break

    cv.destroyAllWindows()
