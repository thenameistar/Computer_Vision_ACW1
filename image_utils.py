import cv2
import numpy as np
# Mapping of detected signs to their assigned numbers
SIGN_NUMBERS = {
    "roundabout": 1,
    "double bend": 2,
    "dual carriageway ends": 3,
    "traffic lights": 4,
    "roadworks": 5,
    "ducks": 6,
    "turn left": 7,
    "keep left": 8,
    "mini roundabout": 9,
    "one way": 10,
    "warning": 11,
    "give way": 12,
    "no entry": 13,
    "stop": 14,
    "20MPH": 15,
    "30MPH": 16,
    "40MPH": 17,
    "50MPH": 18,
    "national speed limit": 19
}


def normalise_brightness(image):
    """applies histogram equalisation to normalise brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    print(f"🔆 average brightness before: {avg_brightness:.2f}")

    if avg_brightness < 80:
        print("🌑 image is too dark, applying brightness enhancement.")
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    elif avg_brightness > 180:
        print("☀️ image is too bright, reducing brightness.")
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.8, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("✅ brightness normalised.")

    # debugging step
    cv2.imshow("brightness adjustment", result)
    cv2.waitKey(0)

    return result

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """applies Gaussian blur to reduce noise."""
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    print("✅ gaussian blur applied.")

    # debugging step
    cv2.imshow("blurred image", blurred)
    cv2.waitKey(0)

    return blurred

def detect_red_signs(image):
    """Detects red regions in an image using HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range (red has two ranges in HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create red mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2  # Combine both masks

    # Apply mask to image
    red_detected = cv2.bitwise_and(image, image, mask=red_mask)

    return red_mask, red_detected

def apply_otsu_threshold(image):
    """Converts an image to grayscale and applies Otsu's threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh

def apply_adaptive_threshold(image):
    """Applies adaptive thresholding for better edge detection in varying lighting conditions."""
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)


def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Applies Canny Edge Detection on the given image."""
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def detect_color_ratio(image, color):
    """calculates the percentage of the specified color in an image using HSV filtering."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color == "red":
        lower_red1 = np.array([0, 120, 120])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 120])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2  # combine both masks

    elif color == "blue":
        lower_blue = np.array([90, 100, 100])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

    else:
        raise ValueError("invalid color specified. choose 'red' or 'blue'.")

    # calculate the color ratio
    color_ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])

    print(f"✅ {color.capitalize()} color ratio: {color_ratio:.2f}")

    return color_ratio

def detect_color_mask(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color == "red":
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 100, 100])
        upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)  # Merge both red ranges

    elif color == "blue":
        lower = np.array([100, 150, 50])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    return mask

def find_sign_contours(image, min_area=500, max_area=5000, use_convex_hull=True):
    """finds and filters contours based on red mask and adaptive preprocessing."""

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"found {len(contours)} contours before filtering.")

    # filter contours based on area
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    print(f"✅ {len(valid_contours)} valid contours remaining after area filtering.")

    # apply Convex Hull to improve shape detection
    if use_convex_hull:
        valid_contours = [cv2.convexHull(cnt) for cnt in valid_contours]
    return valid_contours

def detect_shape(contour):
    """determines the shape of a detected sign based on contour approximation."""
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    num_sides = len(approx)

    print(f"contour has {num_sides} sides.")

    if num_sides > 15:  
        print("⚠️ too many sides detected, ignoring this shape.")
        return "unknown"

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    # prioritise circle detection
    contour_area = cv2.contourArea(contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    ratio = contour_area / circle_area  # measures how well it fits a circle
    
    print(f"contour area: {contour_area:.2f}, circle area: {circle_area:.2f}, ratio: {ratio:.6f}")
    
    shape_name = "unknown"

    if ratio > 0.7 and 0.85 <= aspect_ratio <= 1.15:
        shape_name = "circle"
    if num_sides == 3:
        shape_name = "triangle"
    elif num_sides == 4:
        shape_name = "square" if 0.85 <= aspect_ratio <= 1.15 else "rectangle"   
    elif num_sides == 8:
        shape_name = "octagon"

    print(f"final shape passed: {shape_name}")        
    return shape_name

def identify_sign(cropped_sign, shape):
    """identifies the sign based on color, shape, and assigns a sign number."""
   
    
    # get segmentation masks for both red and blue colors
    red_mask = detect_color_mask(cropped_sign, "red")
    blue_mask = detect_color_mask(cropped_sign, "blue")

    # calculate colour ratios
    red_ratio = cv2.countNonZero(red_mask) / (cropped_sign.shape[0] * cropped_sign.shape[1])
    blue_ratio = cv2.countNonZero(blue_mask) / (cropped_sign.shape[0] * cropped_sign.shape[1])

    print(f"🔎 shape received for identification: '{shape}'")
    print(f"🔴 red ratio: {red_ratio:.2f}, 🔵 blue ratio: {blue_ratio:.2f}")

    # convert to grayscale 
    gray = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)

    # apply Otsus thresholding
    _, binary_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #apply Canny edge detection
    canny_edges = cv2.Canny(gray, 100, 200)

    # apply Morphological Processing for Otsu
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_thresh = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)
    binary_thresh = cv2.morphologyEx(binary_thresh, cv2.MORPH_OPEN, kernel)

    cv2.imshow(f"Otsu threshold - {shape}", binary_thresh)
    cv2.waitKey(0)

    # compare edge detection results
    otsu_contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canny_contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # determine which method detected more relevant contours
    if len(canny_contours) > len(otsu_contours):
      print("✅ using Canny edge detection for shape analysis.")
      selected_contours = canny_contours
      selected_method = "Canny"
    else:
      print("✅ using Otsu’s thresholding for shape analysis.")
      selected_contours = otsu_contours
      selected_method = "Otsu"

    # debugging
    cv2.imshow(f"{selected_method} threshold - sign", binary_thresh if selected_method == "Otsu" else canny_edges)
    cv2.waitKey(0)

    # extract contours from the thresholded image

    if not selected_contours:
        return "unknown", -1
    
    largest_cnt = max(selected_contours, key=cv2.contourArea)
    shape = detect_shape(largest_cnt)
    
    if shape == "unknown":
        return "unknown", -1

        
    # **RED SIGNS** (Warning, Speed Limits, Stop, Give Way)
    if red_ratio > 0.2:
        if shape == "triangle":
            return "warning", SIGN_NUMBERS["warning"]
        elif shape == "inverted_triangle":
            return "give way", SIGN_NUMBERS["give way"]
        elif shape == "octagon":
            return "stop", SIGN_NUMBERS["stop"]
        elif shape == "circle":
            return "speed limit", SIGN_NUMBERS["50MPH"]  # placeholder 

    # **BLUE SIGNS** (Mandatory Direction)
    elif blue_ratio > 0.2:
        return "turn left", SIGN_NUMBERS["turn left"]  # placeholder for regulatory signs

    return "unknown", -1  # default if no match
