# Part III. Process the video of the table tennis game, locating the ball in each frame (if visible).  
# Locate all frames in which the ball changes direction.  Label each of these as (a) ball bounced on the table, 
# (b) ball hit by player, (c) ball hit the net or (d) ball hit something else.  
# Determine your system accuracy on the table tennis video.

import cv2
import numpy as np

# You can remove or adjust this based on the length of your video
MAX_FRAMES = 1000
LEARNING_RATE = -1
fgbg = cv2.createBackgroundSubtractorMOG2()

# Replace 'path/to/your/video.mp4' with the path to your video file
cap = cv2.VideoCapture('Computer Vision Assignment\Part_III\TableTennis.mp4')

for t in range(MAX_FRAMES):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Break the loop if no frame is captured
    if not ret:
        break

    # Apply MOG
    motion_mask = fgbg.apply(frame, LEARNING_RATE)
    # Get background
    background = fgbg.getBackgroundImage()

    # Display the motion mask and background
    cv2.imshow('background', background)
    cv2.imshow('Motion Mask', motion_mask)

    # Exit
    if cv2.waitKey(1) == ord('e'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
