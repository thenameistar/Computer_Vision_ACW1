import cv2
import numpy as np

def threshold_color(image):
    """applies hsv thresholding to detect red, blue, and white regions in the image."""
    
    # convert image to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define color ranges for red
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # define color range for blue
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])

    # define color range for white (for white-background signs)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    # create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2  # combining both red masks

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    return mask_red, mask_blue, mask_white  # ✅ returns exactly 3 values

def find_sign_contours(mask, min_area=500):
    """finds contours in a given binary mask and filters them based on area."""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter contours based on minimum area (to remove noise)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return valid_contours  # ✅ returns the list of contours

def apply_gaussian_blur(image, kernel_size=(15, 15)):
    """applies gaussian blur to reduce noise while preserving edges."""
    
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    print(f"✅ gaussian blur applied with kernel size: {kernel_size}")
    
    return blurred
