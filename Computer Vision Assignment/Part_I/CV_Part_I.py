import cv2 as cv
import numpy as np
import os

def ConvertToHSV(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)

def ThresholdImageForBalls(img_hsv):
    # Threshold for white balls
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 60, 255])
    mask_white = cv.inRange(img_hsv, lower_white, upper_white)

    # Threshold for orange balls
    lower_orange = np.array([0, 100, 220])
    upper_orange = np.array([20, 255, 255])

    mask_orange = cv.inRange(img_hsv, lower_orange, upper_orange)

    # Combine the masks
    mask = cv.bitwise_or(mask_white, mask_orange)

    # Morphological operations to remove noise and fill small holes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # Erode the image
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    return mask

def FindBalls(mask, img):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    for contour in contours:
        # Calculate circularity
        perimeter = cv.arcLength(contour, True)
        area = cv.contourArea(contour)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # More sensitive circularity filter
        if circularity > 0.4:  # Lowered the threshold for circularity
            (x, y), radius = cv.minEnclosingCircle(contour)
            radius = int(radius)
            
            if radius >= 15 and radius <= 50:  # Only consider circles with radius 30 pixels or more
                center = (int(x), int(y))
                
                # Draw the circle on the image
                cv.circle(img, center, radius, (0, 255, 0), 2)
                cv.circle(img, center, 2, (0, 0, 255), 3)

                # Print the center coordinates of the circle and the diameter
                print(center[0], center[1], 2 * radius)
            
    return img

if __name__ == '__main__':
    img_dir = "Computer Vision Assignment\Part_I\Balls"
    img_files = [f for f in os.listdir(img_dir) if f.startswith('Ball') and f.endswith('.jpg')]
    img_files.sort()

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = cv.imread(img_path)

        if img is None:
            print(f"Error: Image {img_file} not loaded!")
            continue
        
        img_hsv = ConvertToHSV(img)
        mask = ThresholdImageForBalls(img_hsv)
        cv.imshow('mask', mask)

        img_detected = FindBalls(mask, img.copy())

        cv.imshow('detected balls', img_detected)
        key = cv.waitKey(0)
        if key == ord('q'):
            break

    cv.destroyAllWindows()