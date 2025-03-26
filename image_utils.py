import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os

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

# Load templates at module level
TEMPLATES = []
sift = cv2.SIFT_create()
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

for filename in os.listdir(TEMPLATE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(TEMPLATE_DIR, filename)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        label = os.path.splitext(filename)[0]
        TEMPLATES.append((label, kp, des, gray))

def normalise_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_brightness = np.mean(hsv[:, :, 2])
    print(f"🔆 average brightness before: {avg_brightness:.2f}")

    if avg_brightness < 80:
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    elif avg_brightness > 180:
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.8, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("✅ brightness normalised.")

    return result

def detect_colour_signs(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Red Mask ---
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # --- Blue Mask ---
    lower_blue1 = np.array([100, 100, 50])
    upper_blue1 = np.array([130, 255, 255])
    lower_blue2 = np.array([85, 30, 30])
    upper_blue2 = np.array([145, 255, 255])
    blue_mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    blue_mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    blue_mask = cv2.bitwise_or(blue_mask1, blue_mask2)

    # --- Apply same morphological filtering to both ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    def clean_mask(mask, min_area=600):
        # Erode and dilate to reduce noise
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Filter by contour area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(cleaned_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        return cleaned_mask

    red_clean = clean_mask(red_mask)
    blue_clean = clean_mask(blue_mask)
    cv2.imshow("Red Mask Cleaned", red_clean)
    cv2.imshow("Blue Mask Cleaned", blue_clean)
    cv2.waitKey(0)

    return red_clean, blue_clean

def extract_glcm_features(image_gray):
    """
    Extracts GLCM texture features from grayscale image.
    Returns contrast, correlation, energy, and homogeneity.
    """
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return contrast, correlation, energy, homogeneity

def detect_shape(contour, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # 🔐 Solidity check
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0

    # 🔎 Filter small/noisy shapes early
    if area < 200 or solidity < 0.6:
        print(f"⚠️ Ignored: Area {area:.1f}, Solidity {solidity:.2f}")
        return "unknown"

    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    num_sides = len(approx)
    print(f"➕ Shape Metrics → Sides: {num_sides}, Circ: {circularity:.2f}, AR: {aspect_ratio:.2f}")
        
    if 0.3 < circularity < 0.6 and 0.7 < aspect_ratio < 1.3:
        return "circle" 

    if num_sides == 8 and 0.65 < circularity < 0.85:
        return "octagon"

    if 3 <= num_sides <= 14 and 0.2 < circularity < 0.7 and 0.2 < aspect_ratio < 1.6:
        return "triangle"

    if 4 <= num_sides <= 6:
        return "rectangle"

    return "unknown"

def get_stats(roi, L=256):
    """
    Calculate statistical texture features for a given region of interest (roi)
    roi: 2D numpy array (grayscale image)
    L: number of gray levels (default 256)
    Returns: mean, variance, r, skewness, uniformity, entropy
    """
    hist = np.histogram(roi, bins=L, range=(0, L), density=True)
    hist = hist[0].reshape(1, -1)

    mean = 0
    uniformity = 0.0
    entropy = 0.0
    for i in range(0, L):
        mean += float(i) * hist[0][i]
        uniformity += hist[0][i] ** 2
        entropy += -hist[0][i] * np.log2(hist[0][i] + 1e-6)  # Avoid log(0)

    # Moments
    m = np.zeros(5)
    for n in range(0, 5):
        for z1 in range(0, L):
            m[n] += (float(z1) - mean) ** n * hist[0][z1]

    variance = m[2]
    norm_variance = variance / ((L - 1) ** 2)
    r = 1 - (1 / (1 + norm_variance))
    skewness = m[3]
    flatness = m[4]

    return mean, variance, r, skewness, uniformity, entropy
    
def find_sign_contours(mask, min_area=300, max_area=100000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"🔍 Found {len(contours)} contours before filtering.")

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull_area if hull_area > 0 else 0
        print(f"➕ Area: {area:.1f}, Circ: {circularity:.2f}, Sol: {solidity:.2f}, AR: {aspect_ratio:.2f}")


        if solidity > 0.6 and area > 600 and 0.3 < aspect_ratio < 2.5:
            valid_contours.append(cnt)
    
    print(f"✅ {len(valid_contours)} valid contours remaining after filtering.")
    return valid_contours


def identify_sign(image, contour, shape, combined_mask):

    # Fallback shape override
    if shape == "unknown":
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.7 < aspect_ratio < 1.3 and circularity > 0.3:
            print("⚠️ Shape fallback override: treating as 'circle'")
            shape = "circle"

    # Crop bounding box of detected sign
    x, y, w, h = cv2.boundingRect(contour)
    cropped_sign = image[y:y+h, x:x+w]
    cropped_mask = combined_mask[y:y+h, x:x+w]

    # Calculate red ratio
    red_area = cv2.countNonZero(cropped_mask)
    roi_area = w * h
    red_ratio = red_area / roi_area if roi_area > 0 else 0
    print(f"🔎 Shape: '{shape}', 🔴 Local Red Ratio: {red_ratio:.2f}")

    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 📊 GLCM Texture Features
    contrast, correlation, energy, homogeneity = extract_glcm_features(gray_blurred)
    print(f"📊 GLCM → Contrast: {contrast:.2f}, Correlation: {correlation:.2f}, Energy: {energy:.4f}, Homogeneity: {homogeneity:.2f}")

    if energy < 0.025 or homogeneity < 0.22:
        mean, var, r, skew, uniformity, entropy = get_stats(gray)
        print(f"📊 Backup Stats → Entropy: {entropy:.2f}, Uniformity: {uniformity:.2f}")
        if entropy < 1.2 or uniformity > 0.9:
            print("⚠️ Both texture checks suggest noise.")
            return "unknown", -1

    if red_ratio < 0.05:
        print("⚠️ Very low red ratio (likely not red).")     
        
    # Try SIFT template matching as a fallback
    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(gray, None)

    if des2 is None:
        print("⚠️ No SIFT features found in candidate sign.")
        return "unknown", -1

    bf = cv2.BFMatcher()
    best_match = None
    best_score = 0

    for label, kp1, des1, _ in TEMPLATES:
        if des1 is None:
            continue
        matches = bf.knnMatch(des1, des2, k=2)
        # Lowe's ratio test
        good = [m for match in matches if len(match) == 2 for m, n in [match] if m.distance < 0.6 * n.distance]

        if len(good) > best_score:
            best_score = len(good)
            best_match = label

    if best_match and best_score > 10:  # You can tune this threshold
        print(f"🧠 SIFT matched: {best_match} with {best_score} good matches.")
        return best_match, SIGN_NUMBERS.get(best_match, -1)

    print("⚠️ Red sign detected but shape/content unclear.")
    return "unknown", -1