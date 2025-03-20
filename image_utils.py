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

DEBUG = False

def normalise_brightness(image):
    """Applies histogram equalization to normalize brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    print(f"🔆 Average brightness before: {avg_brightness:.2f}")

    if avg_brightness < 80:
        print("🌑 Image is too dark, applying brightness enhancement.")
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    elif avg_brightness > 180:
        print("☀️ Image is too bright, reducing brightness.")
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.8, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("✅ Brightness normalized.")

    # Debugging step
    if DEBUG:
        cv2.imshow("Brightness Adjustment", result)
        cv2.waitKey(0)

    return result

def detect_red_signs(image):
    """Detects red regions in an image using HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjusted Red Color Range
    lower_red1 = np.array([0, 50, 50])  # Lowered saturation & value for better detection
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create red mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2  # Combine both masks

    # Apply Morphological Operations to refine mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)   # Remove noise

    if DEBUG:
        cv2.imshow("Debug - Adjusted Red Mask", red_mask)
        cv2.waitKey(0)

    return red_mask

def find_sign_contours(image, min_area=300, max_area=100000, use_convex_hull=False):
    """Finds and filters contours based on red mask and preprocessing."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"🔍 Found {len(contours)} contours before filtering.")

    # Filter contours based on area
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    print(f"✅ {len(valid_contours)} valid contours remaining after area filtering.")

    # Optionally apply Convex Hull
    if use_convex_hull:
        valid_contours = [cv2.convexHull(cnt) for cnt in valid_contours]
    
    return valid_contours

def detect_circles(image):
    """Detects circular shapes using HoughCircles and returns the detected circles."""
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=20, maxRadius=150
    )

    # If circles are detected, return them
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0]  # Return detected circles

    return []  # Return empty list if no circles are found

def detect_shape(contour, image):
    """Determines the shape of a detected sign based on contour approximation or HoughCircles."""
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    num_sides = len(approx)

    # Check for circles separately
    circles = detect_circles(image)
    if len(circles) > 0:
        return "circle"

    if num_sides == 3:
        return "triangle"
    elif 4 <= num_sides <= 6:
        return "rectangle"
    else:
        return "unknown"

def identify_sign(image, contour, shape):
    """Identifies the sign based on color, shape, and assigns a sign number."""
    shape = detect_shape(contour, image)

    # Get red color ratio
    red_mask = detect_red_signs(image)
    red_ratio = cv2.countNonZero(red_mask) / (image.shape[0] * image.shape[1])

    print(f"🔎 Shape received: '{shape}', 🔴 Red ratio: {red_ratio:.2f}")

    if red_ratio > 0.02:
        if shape == "triangle":
            return "warning", SIGN_NUMBERS["warning"]
        elif shape == "circle":
            # Check for different speed limits
            cropped_sign = image
            gray = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)

            # Use Otsu's Threshold for clear segmentation
            _, binary_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours of the numbers
            number_contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If multiple number contours are found, assume it's a speed limit
            if len(number_contours) > 0:
                # TODO: You can use OCR (Tesseract) or pattern matching here
                # Placeholder logic for now:
                if len(number_contours) == 1:
                    return "20MPH", SIGN_NUMBERS["20MPH"]
                elif len(number_contours) == 2:
                    return "30MPH", SIGN_NUMBERS["30MPH"]
                elif len(number_contours) == 3:
                    return "40MPH", SIGN_NUMBERS["40MPH"]
                else:
                    return "50MPH", SIGN_NUMBERS["50MPH"]

            return "speed limit", SIGN_NUMBERS["50MPH"]  # Default speed limit

        elif shape == "octagon":
            return "stop", SIGN_NUMBERS["stop"]

    return "unknown", -1