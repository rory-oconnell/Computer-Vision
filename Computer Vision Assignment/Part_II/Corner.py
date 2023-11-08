import cv2
import numpy as np
import os
import glob

def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def main():
    file_location1 = "Computer Vision Assignment/Part_II/Tables"
    image_files1 = [
        "Table1.jpg",
        "Table2.jpg",
        "Table3.jpg",
        "Table4.jpg",
        "Table5.jpg",
    ]

    images1 = []

    for image_file in image_files1:
        filename = os.path.join(file_location1, image_file)
        image = cv2.imread(filename, -1)
        if image is None:
            print(f"Could not open {filename}")
            return -1
        images1.append(image)

    for current_image in images1:
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        hsv_channels = list(cv2.split(hsv))
        v_channel = hsv_channels[2]  # V channel
        s_channel = hsv_channels[1]  # S channel

        # Apply Histogram Equalization to the V channel
        v_channel = cv2.equalizeHist(v_channel)

        # Adjust the saturation by multiplying the values and clip to ensure pixel value remains between 0 and 255
        saturation_multiplier = 2.2  # Experiment with different values!
        s_channel = np.clip(s_channel.astype('float') * saturation_multiplier, 0, 255).astype('uint8')

        # Merge the adjusted channels back into the HSV image
        hsv_channels[1] = s_channel  # Saturation channel
        hsv_channels[2] = v_channel  # Value channel

        # Convert the list back to a tuple for merging
        hsv = cv2.merge(hsv_channels)

        # Merge the adjusted channels back into the HSV image
        hsv_channels[1] = s_channel
        hsv_channels[2] = v_channel
        hsv = cv2.merge(hsv_channels)

        # Define the range for blue color in HSV
        lower_blue = np.array([90, 50, 50])  # Lower bound for blue color in HSV
        upper_blue = np.array([170, 255, 255])  # Upper bound for blue color in HSV

        # Threshold the HSV image to get only blue colors
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Dilate the blueMask if needed to emphasize the blue regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)

        # Apply the blue color mask to the original image
        blue_filtered = cv2.bitwise_and(current_image, current_image, mask=blue_mask)

        # Rescale the images for display
        current_image_small = rescale_frame(current_image, 0.15)
        blue_filtered_small = rescale_frame(blue_filtered, 0.15)

        # Display the original image
        cv2.imshow("Original Image", current_image_small)

        # Display the result image with the blue color filtered
        cv2.imshow("Blue Color Filtered Image", blue_filtered_small)

        # Wait for a key press and close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
