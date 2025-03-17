import cv2
import os
import numpy as np
from image_utils import (
    normalise_brightness,
    apply_gaussian_blur,
    detect_red_signs,
    apply_adaptive_threshold,
    apply_canny_edge_detection,
    find_sign_contours,
    identify_sign,
    detect_shape
)

DEBUG = True  # Set to False to disable debugging

def process_single_image(image_path, output_filename):
    print(f"\n📂 Processing image: {image_path}")

    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: could not load image {image_path}")
        return
    print("✅ Image loaded successfully.")
    
    # Step 1: Detect Red Signs (HSV Masking)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("Debug - HSV Image", hsv)
    cv2.waitKey(0)

    # Step 2: Detect Red Signs (HSV Masking)
    red_mask, red_detected = detect_red_signs(image)
    cv2.imshow("Debug - Raw Red Mask", red_mask)
    cv2.waitKey(0)

    # Step 3: Apply Morphological Operations to Refine Red Mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)   # Remove noise

    cv2.imshow("Debug - Refined Red Mask", red_mask)
    cv2.waitKey(0)

    # Step 4: Normalize Brightness
    normalized_image = normalise_brightness(image)
    cv2.imshow("Debug - Normalized Brightness", normalized_image)
    cv2.waitKey(0)

    # Step 5: Convert to Grayscale (for Otsu or Canny)
    gray = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Debug - Grayscale Image", gray)
    cv2.waitKey(0)

    # Step 6: Apply Gaussian Blur
    blurred = apply_gaussian_blur(gray, (7, 7))
    cv2.imshow("Debug - Blurred Image", blurred)
    cv2.waitKey(0)

   # Step 7: Choose Processing Method Based on Contrast
    contrast = np.std(gray)
    print(f"📊 Image contrast: {contrast:.2f}")

    if contrast > 50:
        print("✅ Using Canny Edge Detection")
        edges = cv2.Canny(blurred, 100, 200)

         # Apply morphological closing to fill gaps in the edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    else:
        print("✅ Using Adaptive Thresholding")
        edges = apply_adaptive_threshold(blurred)  # Use adaptive thresholding

    cv2.imshow("Debug - Final Edge Detection Output", edges)
    cv2.waitKey(0)

    # Step 8: Combine Edge Detection with Red Mask
    combined_edges = cv2.bitwise_or(edges, red_mask)
    cv2.imshow("Debug - Combined Mask & Edge Detection", combined_edges)
    cv2.waitKey(0)

    # Step 9: Find & Filter Contours
    contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if 200 < cv2.contourArea(cnt) < 60000]
    print(f"🔍 Found {len(contours)} contours before filtering.")
    print(f"✅ {len(valid_contours)} valid contours remaining after area filtering.")

    if not valid_contours:
        print("⚠️ No valid contours detected.")
        return

    # Step 10: Identify Shapes & Recognize Signs
    for idx, cnt in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Ensure bounding box does not go out of image bounds
        x, y, w, h = max(0, x), max(0, y), min(w, image.shape[1] - x), min(h, image.shape[0] - y)
        
        # Crop the detected sign
        cropped_sign = image[y:y+h, x:x+w]
        cv2.imshow(f"Debug - Cropped Sign {idx}", cropped_sign)
        cv2.waitKey(0)

        # Convert to grayscale (For OCR / Shape Detection)
        gray_cropped = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding to enhance contrast
        _, binary_cropped = cv2.threshold(gray_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow(f"Debug - Cropped Threshold {idx}", binary_cropped)
        cv2.waitKey(0)

        shape = detect_shape(cnt)
        sign_name, sign_number = identify_sign(cropped_sign, shape)

        if sign_name != "unknown":
            print(f"✅ Detected: {sign_name} (#{sign_number}) - Shape: {shape}")

            # Normalize Bounding Box Coordinates
            img_height, img_width = image.shape[:2]
            bb_xcentre = (x + w / 2) / img_width
            bb_ycentre = (y + h / 2) / img_height
            bb_width = w / img_width
            bb_height = h / img_height

            # Format Output
            with open(output_filename, "a") as file:
                try:
                    output_data = f"{image_path}, {idx}, {sign_name}, {bb_xcentre:.6f}, {bb_ycentre:.6f}, {bb_width:.6f}, {bb_height:.6f}, 0, 0, 1.0\n" 
                    file.write(output_data)
                    print(f"✅ Successfully wrote to output file: {output_data.strip()}")
                except Exception as e:
                    print(f"❌ ERROR WRITING TO FILE: {e}")

    print("✅ Image processing complete.")
