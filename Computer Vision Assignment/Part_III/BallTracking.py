import cv2

# Initialize video capture from the first camera device
cap = cv2.VideoCapture('Computer Vision Assignment/Part_III/TableTennis.mp4')

# Check if the video capture has been initialized correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Create Background Subtractor MOG2 object
fgbg = cv2.createBackgroundSubtractorMOG2(500, 16, True)

# Set the desired frames per second
FPS = 60

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Start processing
    fgmask = fgbg.apply(frame)

    # Display the foreground mask
    cv2.imshow('Foreground Mask', fgmask)

    # Calculate the time to wait until the next frame to maintain the desired FPS
    delay = int(1000 / FPS)
    key = cv2.waitKey(delay) & 0xFF

    # Exit loop if 'e' is pressed
    if key == ord('e'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()