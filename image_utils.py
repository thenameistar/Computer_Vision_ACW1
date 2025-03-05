import cv2
import numpy as np
import os

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

    return mask_red, mask_blue, mask_white  # returns exactly 3 values

def find_sign_contours(mask, min_area=500):
    """finds contours in a given binary mask and filters them based on area."""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter contours based on minimum area (to remove noise)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return valid_contours  # returns the list of contours

def apply_gaussian_blur(image, kernel_size=(15, 15)):
    """applies gaussian blur to reduce noise while preserving edges."""
    
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    print(f"✅ gaussian blur applied with kernel size: {kernel_size}")
    
    return blurred


def apply_canny_edge_detection(gray_image):
    """applies Canny edge detection for more precise digit edges."""
    return cv2.Canny(gray_image, 50, 150)  # tune thresholds


def apply_otsu_threshold(gray_image):
    """applies Otsu's thresholding to binarise the image."""
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_image


def extract_number_from_sign(cropped_sign):
    """extracts numbers from a speed limit sign using Otsu's thresholding, morphology, and contour filtering."""
    
    print("🟢 extracting numbers from the cropped sign...")  # debugging print

    # convert to grayscale
    gray = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)

    # apply Otsu's thresholding
    binary_image = apply_otsu_threshold(gray)

    # apply morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  # close gaps
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)   # remove small noise

    # apply Canny edge detection for sharper contours
    edge_image = cv2.Canny(binary_image, 50, 150)

    # find contours in the processed image
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # debugging: show all contours before filtering
    debug_image = cropped_sign.copy()
    cv2.drawContours(debug_image, contours, -1, (255, 0, 0), 2)  # blue contours
    cv2.imshow("all contours before filtering", debug_image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    # sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # if no contours are found, return 0
    if len(contours) == 0:
        print("❌ no contours found in edge-detected image!")
        return 0  

    # assume the largest contour is the outer circle (ignore it)
    largest_contour = contours[0]

    digit_contours = []
    for cnt in contours:
        if np.array_equal(cnt, largest_contour):  # ignore the largest contour
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        # digits should be within a reasonable area and aspect ratio
        if 30 < area < 5000 and 0.1 < aspect_ratio < 2.0:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(cropped_sign, [hull], -1, (0, 255, 0), 2)
            digit_contours.append(cnt)

    # debugging output
    print(f"🟢 detected {len(digit_contours)} possible digits")

    # save detected digits image
    cv2.imwrite("results/detected_numbers.jpg", cropped_sign)

    # show debugging images 
    cv2.imshow("Thresholded Image", binary_image)
    cv2.imshow("Edge Detected Image", edge_image)
    cv2.imshow("Detected Numbers", cropped_sign)
    cv2.waitKey(0)  # auto-close after 500ms
    cv2.destroyAllWindows()
    
    return len(digit_contours)  # return number of detected digits
