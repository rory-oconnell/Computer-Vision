# Part III. Process the video of the table tennis game, locating the ball in each frame (if visible).  
# Locate all frames in which the ball changes direction.  
# Label each of these as (a) ball bounced on the table, 
# (b) ball hit by player, (c) ball hit the net or (d) ball hit something else.  
# Determine your system accuracy on the table tennis video.
import cv2
import numpy as np

MAX_FRAMES = 100000
LEARNING_RATE = -1
count = 0
fgbg = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture('Computer Vision Assignment/Part_III/TableTennis.mp4')

# Define the lower and upper bounds for the orange color in HSV space
lower_bound = np.array([0, 100, 220])
upper_bound = np.array([20, 255, 255])

prev_ball_x_state = None  # Store the previous ball state
prev_ball_y_state = None  # Store the previous ball state

x_centre_prev = None  # Store the previous centre x-coordinate
y_centre_prev = None  # Store the previous centre y-coordinate

x_state_cur = None  # Store the current ball state
y_state_cur = None  # Store the current ball state

# Define size thresholds for the ball detection
MIN_CONTOUR_AREA = 50  # Minimum area of the contour to be considered as the ball
MAX_CONTOUR_AREA = 500  # Maximum area of the contour to be considered as the ball

for t in range(MAX_FRAMES):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Define the region for the net
    left_net = (frame.shape[1] // 2) - 15
    right_net = (frame.shape[1] // 2) + 15
    
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

            centre_x = x + w//2  # Calculate the X coordinate of the centre
            centre_y = y + h//2  # Calculate the Y coordinate of the centre
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Ball', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Check if the ball is moving left or right
            if x_centre_prev is not None:
                if centre_x < x_centre_prev:
                    x_state_cur = "left"
                    # print(x_state_cur)
                    
                elif centre_x > x_centre_prev:
                    x_state_cur = "right"
                    # print(x_state_cur)

            if y_centre_prev is not None:
                if centre_y < y_centre_prev:
                    y_state_cur = "up"
                    # print(y_state_cur)
                    
                elif centre_y > y_centre_prev:
                    y_state_cur = "down"
                    # print(y_state_cur)     
            
            # Check if the ball has changed direction
            if x_state_cur != prev_ball_x_state:
                if x_centre_prev is not None and abs(centre_x - x_centre_prev) > 15:
                    NetFlag = False
                    print(f"Frame {t} ({centre_x}, {centre_y}) Hit by Player")

            # Check if the ball has changed direction
            elif y_state_cur == "up" and prev_ball_y_state == "down" and NetFlag == False:
                print(f"Frame {t} ({centre_x}, {centre_y}) Bounce on Table")

            # Check if the ball has hit the net
            elif x_centre_prev is not None and abs(centre_x - x_centre_prev) < 6 and centre_x > left_net and centre_x < right_net and NetFlag == False:
                print(f"Frame {t} ({centre_x}, {centre_y}) Hit the Net")
                NetFlag = True

            # print(centre_x, centre_y)
            x_centre_prev = centre_x  # Store the current centre x-coordinate
            y_centre_prev = centre_y
            prev_ball_x_state = x_state_cur
            prev_ball_y_state = y_state_cur


    # Display the original frame with the potential ball highlighted
    cv2.imshow('Frame with Ball', frame)
    
    # Wait for a key press to continue to the next frame
    key = cv2.waitKey(10)  # Delay of 30ms between frames
    
    # Exit on 'e' key press or if frame read was unsuccessful
    if key == ord('e') or not ret:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
