import argparse
import os
import image_processing
import video_processing


def main():
    print("✅ Script started")  # Verifying execution

    parser = argparse.ArgumentParser(description="Traffic Sign Detection")
    parser.add_argument("--image", help="Path to a single image file")
    parser.add_argument("--inputfile", help="Path to a text file with an image list")
    parser.add_argument("--video", help="Path to a video file (or 'webcam' for live feed)")
    parser.add_argument("--output", help="Path to output file for detection results", default="output.txt")

    args = parser.parse_args()  # ✅ FIX: Parse arguments first

    # Ensure output file is set correctly
    if not args.output:
        args.output = "output.txt"

    # Ensure output file is cleared before writing new results
    with open(args.output, "w") as file:
        file.write("")  # Clears previous data

    if args.image:
        image_path = os.path.join("acw1_test_images", args.image)  # Ensure correct path
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file '{image_path}' not found.")
            return
        print(f"📂 Processing single image: {image_path}")
        image_processing.process_single_image(image_path, args.output)

    elif args.inputfile:
        if not os.path.exists(args.inputfile):
            print(f"❌ Error: Input file '{args.inputfile}' not found.")
            return

        with open(args.inputfile, "r") as file:
            image_list = file.read().splitlines()

        print(f"📄 Processing multiple images from: {args.inputfile}")
        for image_name in image_list:
            image_path = os.path.join("acw1_test_images", image_name)
            if os.path.exists(image_path):
                print(f"🔍 Processing: {image_path}")
                image_processing.process_single_image(image_path, args.output)
            else:
                print(f"⚠️ Skipping '{image_path}' (file not found)")

    elif args.video:
        print(f"📹 Processing video: {args.video}")
        video_processing.process_video(args.video, args.output)

    else:
        print("❌ Please provide --image, --inputfile, or --video.")

if __name__ == "__main__":
    main()