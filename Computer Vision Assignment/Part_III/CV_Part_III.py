# Part III. Process the video of the table tennis game, locating the ball in each frame (if visible).  
# Locate all frames in which the ball changes direction.  Label each of these as (a) ball bounced on the table, 
# (b) ball hit by player, (c) ball hit the net or (d) ball hit something else.  
# Determine your system accuracy on the table tennis video.
import cv2
import numpy as np

MAX_FRAMES = 1000
counter = 0
LEARNING_RATE = -1
fgbg = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture('Computer Vision Assignment/Part_III/TableTennis.mp4')

# Define the lower and upper bounds for the orange color in HSV space
lower_bound = np.array([0, 100, 220])
upper_bound = np.array([20, 255, 255])

# Define size thresholds for the ball detection
MIN_CONTOUR_AREA = 50  # Minimum area of the contour to be considered as the ball
MAX_CONTOUR_AREA = 500  # Maximum area of the contour to be considered as the ball

for t in range(MAX_FRAMES):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Calculate the x-coordinate of the center of the frame
    center_x = frame.shape[1] // 2
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply MOG 
    motion_mask = fgbg.apply(frame, LEARNING_RATE)
    
    # Find contours in the motion mask
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Only consider contours within the size range to be the ball
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        # Create a mask for the current contour
        contour_mask = np.zeros_like(motion_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # Use the mask to extract the potential ball's color from the original image
        ball_hsv = cv2.bitwise_and(hsv, hsv, mask=contour_mask)

        # Check if the color of the detected object falls within the desired color range
        ball_color_mask = cv2.inRange(ball_hsv, lower_bound, upper_bound)
        
        # If there's a match, draw a bounding box around the potential ball in the original frame
        if cv2.countNonZero(ball_color_mask) > 0:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w//2  # Calculate the X coordinate of the center
            center_y = y + h//2  # Calculate the Y coordinate of the center
            counter += 1
            print(f"Ball center location: ({center_x}, {center_y}, {counter})")  # Print the center location
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Ball', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the original frame with the potential ball highlighted
    cv2.imshow('Frame with Ball', frame)
    
    # Wait for a key press to continue to the next frame
    key = cv2.waitKey(30)  # Delay of 30ms between frames
    
    # Exit on 'e' key press or if frame read was unsuccessful
    if key == ord('e') or not ret:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
