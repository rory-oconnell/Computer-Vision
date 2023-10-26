import cv2 as cv

img = cv.imread('opencv-course\Resources\Photos\cat.jpg')
cv.imshow('Cat', img)

def rescaleFrame(frame, scale=0.75):
    # Works for images, videos and live video

    # frame.shape[1] is the width of the frame
    width = int(frame.shape[1] * scale)

    # frame.shape[0] is the height of the frame
    height = int(frame.shape[0] * scale)

    # Dimensions of the frame
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_image = rescaleFrame(img)
cv.imshow('Image', resized_image)

def changeRes(width, height):
    # Only works for live video
    capture.set(3, width)
    capture.set(4, height)

# Reading Videos
capture = cv.VideoCapture('opencv-course\Resources\Videos\dog.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, scale=0.2)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    # If the d key is pressed, break out of the loop
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)