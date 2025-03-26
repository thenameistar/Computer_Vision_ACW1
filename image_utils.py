import cv2
import numpy as np

sift = cv2.SIFT_create()

def compute_descriptors(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Example templates
templates = {
    "double bend": compute_descriptors("templates/double_bend.jpg"),
    "roadworks": compute_descriptors("templates/roadworks.jpg"),
    "ducks": compute_descriptors("templates/ducks.jpg"),
}

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

def detect_colour_signs(image):
    """Detects red regions in an image using HSV color space with adaptive thresholding and preprocessing."""
    
    # Step 1: Convert to HSV and Normalize Brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # Brightness channel
    avg_brightness = np.mean(v_channel)

    print(f"🔆 Image Brightness: {avg_brightness:.2f}")
   
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

   # ==== BLUE MASK ====
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # ==== Combine Both Masks ====
    combined_mask = cv2.bitwise_or(red_mask, blue_mask)

    # ==== Morphological Cleaning ====
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.medianBlur(combined_mask, 5)

    # ==== Debug View ====
    cv2.imshow("Debug - Combined Red + Blue Mask", combined_mask)
    cv2.waitKey(0)

    return combined_mask

def detect_shape(contour, image):
    """Determines the shape of a detected sign based on contour approximation, circularity, and HoughCircles."""
    
    # Convert to grayscale and blur for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Get bounding box and aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    # **Step 1: Calculate Circularity**
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Step 2: Debugging Info (🔍 use this to understand why shapes fail)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    num_sides = len(approx)
    print(f"➕ Shape Metrics → Sides: {num_sides}, Circ: {circularity:.2f}, AR: {aspect_ratio:.2f}")

    # Step 3: Hough Circle Detection
    if 0.8 < aspect_ratio < 1.2 and circularity > 0.5:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=80, param2=35, minRadius=20, maxRadius=150
        )
        if circles is not None:
            return "circle"

    # Step 4: Octagon
    if num_sides == 8 and 0.65 < circularity < 0.85:
        return "octagon"

    # Step 5: Triangle (relaxed detection for triangle-shaped warning signs)
    if 3 <= num_sides <= 7 and 0.3 < circularity < 0.65 and 0.8 < aspect_ratio < 1.4:
        return "triangle"

    # Step 6: Rectangle
    if 4 <= num_sides <= 6:
        return "rectangle"

    return "unknown"
    
def find_sign_contours(mask, min_area=300, max_area=100000, enforce_circle=False, triangle_friendly=True):
    """
    Finds and filters contours from the red mask.
    - If `enforce_circle` is True, filters only near-circular contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"🔍 Found {len(contours)} contours before filtering.")

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        # Basic shape metrics
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Convexity check (optional)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area > 0 else 0
         # DEBUG: Print key contour stats
        print(f"➕ Area: {area:.1f}, Circ: {circularity:.2f}, Sol: {solidity:.2f}, AR: {aspect_ratio:.2f}")

        # ---- Filtering Conditions ----
        if enforce_circle:
            if 0.9 <= aspect_ratio <= 1.1 and circularity > 0.8 and solidity > 0.85 and min_area < area < max_area:
                valid_contours.append(cnt)
        else:
            # Triangle-friendly logic (relax constraints for hollow signs)
            if triangle_friendly:
                if area > 100 and solidity > 0.6:
                    valid_contours.append(cnt)
            else:
                if min_area < area < max_area and solidity > 0.8:
                    valid_contours.append(cnt)

    print(f"✅ {len(valid_contours)} valid contours remaining after filtering.")
    return valid_contours

def identify_sign(image, contour, shape):
    """Identifies the sign based on shape and internal content, and assigns a sign number."""
    
    combined_mask = detect_colour_signs(image)

    # Crop bounding box of detected sign
    x, y, w, h = cv2.boundingRect(contour)
    cropped_sign = image[y:y+h, x:x+w]
    cropped_mask = combined_mask[y:y+h, x:x+w]

    # Calculate red ratio inside the sign's bounding box (localized)
    red_area = cv2.countNonZero(cropped_mask)
    roi_area = w * h
    red_ratio = red_area / roi_area if roi_area > 0 else 0

    print(f"🔎 Shape: '{shape}', 🔴 Local Red Ratio: {red_ratio:.2f}")

    if red_ratio > 0.02:
    # ✅ TRIANGLE SIGNS
        if shape == "triangle":
            return "double bend", SIGN_NUMBERS["double bend"]
    # ✅ STOP SIGN (Red + Octagon)
        if shape == "octagon":
            return "stop", SIGN_NUMBERS["stop"]

        # ✅ Speed Limits or No Entry (Red + Circle)
        if shape == "circle":
            # Convert cropped sign to grayscale and binarize
            gray = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)
            _, binary_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Detect white regions (digits or bar)
            number_contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            for cnt in number_contours:
                area = cv2.contourArea(cnt)
                x_, y_, w_, h_ = cv2.boundingRect(cnt)
                if area > 100:  # filter small noise
                    valid_contours.append(cnt)

            # ✅ NO ENTRY Detection (One horizontal bar)
            if len(valid_contours) == 1:
                x_, y_, w_, h_ = cv2.boundingRect(valid_contours[0])
                aspect_ratio = w_ / h_
                sign_center_y = cropped_sign.shape[0] // 2
                bar_center_y = y_ + (h_ // 2)
                vertical_position = abs(sign_center_y - bar_center_y) / h_

                if aspect_ratio > 2.5 and vertical_position < 0.5:
                    return "no entry", SIGN_NUMBERS["no entry"]

            # ✅ SPEED LIMIT (Multiple digits)
            num_digits = len(valid_contours)
            print(f"🔢 Found {num_digits} valid digit/bar contours inside circle.")

            if num_digits == 1:
                return "20MPH", SIGN_NUMBERS["20MPH"]
            elif num_digits == 2:
                return "30MPH", SIGN_NUMBERS["30MPH"]
            elif num_digits == 3:
                return "40MPH", SIGN_NUMBERS["40MPH"]
            else:
                return "50MPH", SIGN_NUMBERS["50MPH"]

        # Fallback warning
        print("⚠️ Red sign detected but shape/content unclear.")

    return "unknown", -1