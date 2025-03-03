import cv2
import numpy as np
import os
from image_utils import threshold_color, find_sign_contours, apply_gaussian_blur

print("📂 saving files to results/ folder...")  # initial debug to check execution

def process_single_image(image_path, output_filename):
    # load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ error: could not load image from {image_path}")
        return

    # ensure 'results/' directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # update output file path to save inside 'results/' folder
    output_filename = os.path.join(results_dir, os.path.basename(output_filename))

    # apply gaussian blur for noise reduction
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    print("✅ gaussian blur applied!")

    # compare original vs blurred image
    comparison = np.hstack((image, blurred_image))
    cv2.imshow("original vs blurred", comparison)
    cv2.imwrite(os.path.join(results_dir, "blurred_image.jpg"), blurred_image)
    print("✅ blurred image saved!")
    cv2.waitKey(0)  

    # apply thresholding function
    mask_red, mask_blue, mask_white = threshold_color(blurred_image)
    print(f"🟥 red mask: {np.sum(mask_red)} pixels")
    print(f"🟦 blue mask: {np.sum(mask_blue)} pixels")
    print(f"⬜ white mask: {np.sum(mask_white)} pixels")

    # find contours for red, blue, and white masks
    contours_red = find_sign_contours(mask_red)
    contours_blue = find_sign_contours(mask_blue)
    contours_white = find_sign_contours(mask_white)

    # draw contours on the original image
    output_image = image.copy()
    cv2.drawContours(output_image, contours_red, -1, (0, 0, 255), 2)  
    cv2.drawContours(output_image, contours_blue, -1, (255, 0, 0), 2)  
    cv2.drawContours(output_image, contours_white, -1, (255, 255, 255), 2)  

    # save the final processed image
    cv2.imwrite(os.path.join(results_dir, "detected_signs.jpg"), output_image)
    print("✅ detected signs image saved!")

    # crop the detected sign from the image (if any red contours found)
    if contours_red:
        x, y, w, h = cv2.boundingRect(contours_red[0])  
        cropped_sign = image[y:y+h, x:x+w]
        
        # save the cropped sign
        cropped_sign_path = os.path.join(results_dir, "cropped_sign.jpg")
        cv2.imwrite(cropped_sign_path, cropped_sign)
        print(f"✅ cropped sign saved to {cropped_sign_path}")

    # show the final detection result
    cv2.imshow("detected Signs", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # write detection results to the output file
    with open(output_filename, "w") as f:
        f.write("Traffic Sign Detection Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"image processed: {image_path}\n")
        f.write(f"red contours detected: {len(contours_red)}\n")
        f.write(f"blue contours detected: {len(contours_blue)}\n")
        f.write(f"white contours detected: {len(contours_white)}\n")
        f.write("=" * 40 + "\n")

    print(f"✅ results saved to {output_filename}")
