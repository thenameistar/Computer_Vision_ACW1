import cv2
import numpy as np

# Mapping of detected signs to assigned numbers
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
    """Normalizes brightness using histogram equalization."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    print(f"🔆 Average brightness before: {avg_brightness:.2f}")

    if avg_brightness < 80:
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    elif avg_brightness > 180:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.8, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("✅ Brightness normalized.")

    # Always show debug image
    cv2.imshow("Brightness Adjustment", result)
    cv2.waitKey(0)

    return result

def detect_red_signs(image):
    """Detects red regions in an image using HSV color space with adaptive thresholding."""
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate mean brightness (V-channel)
    brightness = np.mean(hsv[:, :, 2])
    
    # Adjust red detection thresholds dynamically based on brightness
    if brightness < 100:  # Dark images
        lower_red1 = np.array([0, 80, 40])   
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 40])
        upper_red2 = np.array([180, 255, 255])
    elif brightness < 150:  # Mid-brightness images
        lower_red1 = np.array([0, 90, 50])    
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 90, 50])
        upper_red2 = np.array([180, 255, 255])
    else:  # Bright images
        lower_red1 = np.array([0, 100, 70])   
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 70])
        upper_red2 = np.array([180, 255, 255])
    
    # Create red mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2  

    # Morphological Operations for noise reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Debugging
    cv2.imshow("Debug - Adjusted Red Mask", red_mask)
    cv2.waitKey(0)

    return red_mask

def detect_shape(contour, image):
    """Determines the shape of a detected sign based on contour approximation or HoughCircles."""

    # Convert to grayscale and apply Gaussian Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    approx = cv2.approxPolyDP(contour, 0.015 * cv2.arcLength(contour, True), True)
    num_sides = len(approx)

    if num_sides == 8:
        return "octagon"
    elif num_sides == 3:
        return "triangle"
    elif 4 <= num_sides <= 6:
        return "rectangle"

    # Attempt to detect circles
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=80, param2=40, minRadius=20, maxRadius=150
    )

    if circles is not None:
        for (x, y, r) in circles[0]:
            # Get the bounding box and aspect ratio check
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Ensure the aspect ratio is nearly 1:1 and area ratio matches
            contour_area = cv2.contourArea(contour)
            circle_area = np.pi * (r ** 2)
            
            if 0.9 < aspect_ratio < 1.1 and 0.75 < (contour_area / circle_area) < 1.25:
                return "circle"

    return "unknown"  # Default if no match
    
def find_sign_contours(image, min_area=300, max_area=100000, enforce_circle=False):
    """Finds and filters contours based on red mask and preprocessing.
       - `enforce_circle=True` ensures only circular signs are detected.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"🔍 Found {len(contours)} contours before filtering.")

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
            
            if enforce_circle:
                # **Only keep circular signs**
                if 0.85 <= aspect_ratio <= 1.15 and circularity > 0.7:
                    valid_contours.append(cnt)
            else:
                # **Allow all shapes through**
                valid_contours.append(cnt)

    print(f"✅ {len(valid_contours)} valid contours remaining after filtering.")
    return valid_contours

def identify_sign(image, contour, shape):
    """Identifies the sign based on color, shape, and assigns a sign number."""
    
    shape = detect_shape(contour, image)

    # Get red color ratio
    red_mask = detect_red_signs(image)
    red_ratio = cv2.countNonZero(red_mask) / (image.shape[0] * image.shape[1])

    print(f"🔎 Shape: '{shape}', 🔴 Red ratio: {red_ratio:.2f}")

    if red_ratio > 0.02:
        # **Octagon Detection for STOP Sign**
        if shape == "octagon":
            return "stop", SIGN_NUMBERS["stop"]

        # **Speed Limit & No Entry Logic**
        if shape == "circle":
            # Extract bounding box of detected sign
            x, y, w, h = cv2.boundingRect(contour)
            cropped_sign = image[y:y+h, x:x+w]  # Crop detected sign

            # Convert to grayscale and apply Otsu's Thresholding
            gray = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)
            _, binary_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours of internal white regions (digits or horizontal bar)
            number_contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # **Filter Valid Contours (Ignore Noise)**
            valid_contours = []
            for cnt in number_contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h  # Width-to-height ratio

                # Ignore very small contours (noise)
                if area < 100:
                    continue
                
                # **"No Entry" Sign Detection (Wide Horizontal Bar)**
                if aspect_ratio > 3.5:  # Wide bar with high width/height ratio
                    return "no entry", SIGN_NUMBERS["no entry"]

                # Otherwise, add to valid digit contours
                valid_contours.append(cnt)

            # **Speed Limit Classification Based on Digit Count**
            num_digits = len(valid_contours)
            print(f"🔢 Found {num_digits} valid digit contours.")

            if num_digits == 1:
                return "20MPH", SIGN_NUMBERS["20MPH"]
            elif num_digits == 2:
                return "30MPH", SIGN_NUMBERS["30MPH"]
            elif num_digits == 3:
                return "40MPH", SIGN_NUMBERS["40MPH"]
            else:
                return "50MPH", SIGN_NUMBERS["50MPH"]  # Default

    return "unknown", -1  # Return unknown if no match found