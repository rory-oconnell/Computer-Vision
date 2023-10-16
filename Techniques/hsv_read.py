import cv2
import numpy as np

lower = np.array([160, 100, 20])
upper = np.array([255, 255, 255])

# image = cv2.imread("hsv.png")
image = cv2.imread("Ball1.jpg")
mask = cv2.inRange(image, lower, upper)

cv2.imshow('image', image)
cv2.imshow('mask', mask)

cv2.waitKey(0)

# yellowRoadAvg_RGB = rgb(209, 171, 18)
# yellowRoadAvg_HSV = hsv(48, 91.4, 82)

# lightestYellow_RGB = rgb(237, 200, 4)
# darkestYellow = rgb(177, 145, 29)

# darkestYellow_HSV = hsv(45, 83.1, 69.4)
# lightestYellow_HSV = hsv(50, 98.3, 92.9)


# pingPongBallAvg_RGB = rgb(250, 232, 203)
# pingPongBallAvg_HSV = hsv(23, 18.8, 98)

# pingPongBallLightest_RGB = rgb(254, 255, 254)
# pingPongBallDarkest_RGB = rgb(241, 186, 138)

# pingPongBallLightest_HSV = hsv(85, 0.4, 99.6)
# pingPongBallDarkest_HSV = hsv(23, 42.3, 94.5)
