import cv2 as cv

# Reading Images
#img = cv.imread('opencv-course\Resources\Photos\cat.jpg')

# Displays the image as a new window
#cv.imshow('Cat', img)

# Reading Videos
capture = cv.VideoCapture('opencv-course\Resources\Videos\dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    # If the d key is pressed, break out of the loop
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

# -215 Assertion failed error - could not find the file. The video ended, no more frames to read
# Same will happen with a webcam if the webcam is not connected
# Or with a picture if the path is incorrect

# Waits for a key press to close the window, 0 means infinite time
cv.waitKey(0)