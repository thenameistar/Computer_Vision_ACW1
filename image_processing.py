import cv2
import numpy as np
import os
from image_utils import threshold_color, find_sign_contours, apply_gaussian_blur, extract_number_from_sign

def process_single_image(image_path, output_filename):
    print("📂 starting image processing...")  

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ error: Could not load image from {image_path}")
        return

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # apply Gaussian blur
    blurred_image = apply_gaussian_blur(image)
    print("✅ gaussian blur applied!")
    cv2.imshow("blurred image", blurred_image)
    cv2.imwrite(os.path.join(results_dir, "blurred_image.jpg"), blurred_image)
    cv2.waitKey(500)

    # apply thresholding and find contours
    mask_red, mask_blue, mask_white = threshold_color(blurred_image)
    contours_red = find_sign_contours(mask_red)

    # draw and save detected contours
    output_image = image.copy()
    cv2.drawContours(output_image, contours_red, -1, (0, 0, 255), 2)
    cv2.imshow("detected signs", output_image)
    cv2.imwrite(os.path.join(results_dir, "detected_signs.jpg"), output_image)
    print("✅ detected signs image saved!")
    cv2.waitKey(500)

    if contours_red:
        x, y, w, h = cv2.boundingRect(contours_red[0])
        cropped_sign = image[y:y+h, x:x+w]

        cropped_sign_path = os.path.join(results_dir, "cropped_sign.jpg")
        cv2.imshow("cropped sign", cropped_sign)
        cv2.imwrite(cropped_sign_path, cropped_sign)
        print(f"✅ cropped sign saved to {cropped_sign_path}")
        cv2.waitKey(500)

        # extract digits from the sign
        num_digits = extract_number_from_sign(cropped_sign)
        print(f"🔢 detected {num_digits} digits in the sign")
    
    cv2.destroyAllWindows()