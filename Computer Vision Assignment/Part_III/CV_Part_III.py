# Part III. Process the video of the table tennis game, locating the ball in each frame (if visible).  
# Locate all frames in which the ball changes direction.  Label each of these as (a) ball bounced on the table, 
# (b) ball hit by player, (c) ball hit the net or (d) ball hit something else.  
# Determine your system accuracy on the table tennis video.

import cv2
import numpy as np

cap = cv2.VideoCapture("Computer Vision Assignment\Part_III\TableTennis.mp4")

while True:
    _, frame = cap.read()

    cv2.imshow('Frame', frame)
    cv2.waitKey(0)