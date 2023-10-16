# import the required library
import cv2

# read the input image
image = cv2.imread('Ball5.jpg')
cv2.imshow('original', image)

# Greyscale the image
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Gaussian blur the image
grey = cv2.GaussianBlur(grey, (11, 11), cv2.BORDER_DEFAULT)


# define the alpha and beta
alpha = 1 # Contrast control
beta = 5 # Brightness control

# call convertScaleAbs function
adjusted = cv2.convertScaleAbs(grey, alpha=alpha, beta=beta)

# display the output image
cv2.imshow('adjusted', adjusted)
cv2.waitKey()
cv2.destroyAllWindows()