import argparse
import image_processing
import video_processing

def main():
    print("✅ script started")  # verifying execution

    parser = argparse.ArgumentParser(description="traffic sign detection")
    parser.add_argument("--image", help="path to a single image file")
    parser.add_argument("--inputfile", help="path to a text file with image list")
    parser.add_argument("--video", help="path to a video file (or 'webcam' for live feed)")
    parser.add_argument("--output", help="path to output file for detection results", default="outfile.txt")
    
    args = parser.parse_args()
    
    if args.image:
        image_path = f"acw1_test_images/{args.image}"  # making sure the correct path is used
        image_processing.process_single_image(image_path, args.output)
    elif args.inputfile:
        print("process_multiple_images() is not implemented yet. this will be added later.")
    elif args.video:
        video_processing.process_video(args.video, args.output)
    else:
        print("❌ please provide --image, --inputfile, or --video.")

if __name__ == "__main__":
    main()
