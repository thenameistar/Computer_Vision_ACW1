import cv2
import os
import numpy as np
from image_utils import (
    normalise_brightness,
    detect_red_signs,
    find_sign_contours,
    identify_sign,
    detect_shape
)

DEBUG = True  # Set to False to disable debugging

def process_single_image(image_path, output_filename, visualize=False):
    print(f"\n📂 Processing image: {image_path}")

    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image {image_path}")
        return
    print("✅ Image loaded successfully.")

    # Step 1: Normalize Brightness
    normalized_image = normalise_brightness(image)

    if DEBUG:
        cv2.imshow("Debug - Normalized Brightness", normalized_image)
        cv2.waitKey(0)

    # Step 2: Detect Red Signs (HSV Masking)
    red_mask = detect_red_signs(normalized_image)  # FIXED ✅
    
    if DEBUG:
        cv2.imshow("Debug - Raw Red Mask", red_mask)
        cv2.waitKey(0)

    # Step 3: Find & Filter Contours
    valid_contours = find_sign_contours(red_mask, min_area=300, max_area=100000)
    if not valid_contours:
        print("⚠️ No valid contours detected.")
        return
    
    # Step 4: Identify Shapes & Recognize Signs
    for idx, cnt in enumerate(valid_contours):
        shape = detect_shape(cnt, normalized_image)  # FIXED ✅
        sign_name, sign_number = identify_sign(normalized_image, cnt, shape)  # FIXED ✅

        if sign_name != "unknown":
            print(f"✅ Detected: {sign_name} (#{sign_number}) - Shape: {shape}")

            # Get bounding box & normalize coordinates
            x, y, w, h = cv2.boundingRect(cnt)
            img_height, img_width = image.shape[:2]
            bb_xcentre = (x + w / 2) / img_width
            bb_ycentre = (y + h / 2) / img_height
            bb_width = w / img_width
            bb_height = h / img_height

            # Save Output to File
            with open(output_filename, "a") as file:
                file.write(f"{image_path}, {idx}, {sign_name}, {bb_xcentre:.6f}, {bb_ycentre:.6f}, {bb_width:.6f}, {bb_height:.6f}, 0, 0, 1.0\n")

            # Visualize Detected Bounding Boxes (Only If `visualize=True`)
            if visualize or DEBUG:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, f"{sign_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
               
    if visualize or DEBUG:
        cv2.imshow("Final Detections", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("✅ Image processing complete.")