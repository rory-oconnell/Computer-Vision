import cv2
import numpy as np

# Open the video file or capture device
cap = cv2.VideoCapture('Computer Vision Assignment\Part_III\TableTennis.mp4') # Replace with 0 for webcam

# Check if the video was opened correctly
if not cap.isOpened():
    raise IOError("Cannot open video")

# Read the first frame
ret, prev_frame = cap.read()

if not ret:
    raise IOError("Cannot read video file")

# Convert the first frame to grayscale
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Loop over all the frames in the video
while True:
    # Read the next frame
    ret, curr_frame = cap.read()
    
    if not ret:
        break  # Break the loop if there are no frames left
    
    # Convert current frame to grayscale
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between current frame and the previous frame
    diff = cv2.absdiff(prev_frame_gray, curr_frame_gray)
    
    # Apply a threshold to the difference
    th = 1
    imask = diff > th
    
    # Create a canvas to draw the result
    canvas = np.zeros_like(curr_frame, np.uint8)
    canvas[imask] = curr_frame[imask]
    
    # Show the result
    cv2.imshow('Result', canvas)
    
    # Write the result to a file (if needed)
    # cv2.imwrite("result_frame.png", canvas)
    
    # Set the current frame to previous_frame for the next loop iteration
    prev_frame_gray = curr_frame_gray
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()