import cv2
import numpy as np

def DetectCircles(img):
    # Detecting circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=3, maxRadius=20)
    return circles

MAX_FRAMES = 1000
LEARNING_RATE = -1
fgbg = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture('Computer Vision Assignment/Part_III/TableTennis.mp4')

for t in range(MAX_FRAMES):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Apply MOG 
    motion_mask = fgbg.apply(frame, LEARNING_RATE)
    
    # Get background
    background = fgbg.getBackgroundImage()

    # Converting to grayscale
    circles = DetectCircles(motion_mask)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(motion_mask, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(motion_mask, (i[0], i[1]), 2, (0, 0, 255), 3)
            # Print the center coordinates of the circle and the diameter
            print(i[0], i[1], 2 * i[2])

    
    # Display the motion mask and background
    cv2.imshow('Balls Detected', motion_mask)
    cv2.imshow('Background', background)
    cv2.imshow('Motion Mask', motion_mask)
    
    # Wait for a key press to continue to the next frame
    key = cv2.waitKey(0)
    
    # Exit on 'e' key press or if frame read was unsuccessful
    if key == ord('e') or not ret:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
