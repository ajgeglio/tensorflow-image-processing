import os
import sys
import argparse

# Ensure src is on sys.path regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import Utils
from predictor import Predict

def parse_args():
    parser = argparse.ArgumentParser(description="Substrate Predictor Inference Arguments")
    parser.add_argument("--name", type=str, default="substrate_inference", help="Name for the prediction run")
    parser.add_argument("--start_batch", type=int, default=0, help="Batch index to start from")
    parser.add_argument("--classes", type=str, choices=["2c", "6c"], default="2c", help="Class type: '2c' or '6c'")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--image_list_file", type=str, default=None, help="Text file with list of image paths")
    parser.add_argument("--weights_dir", type=str, default="src\\models", help="Directory containing model weights")
    return parser.parse_args()

def print_warranty():
    warranty_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "warranty")
    try:
        with open(warranty_path, "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        print("Warning: Could not read warranty file.", e)

def main():
    """
    Main entry point for running substrate prediction inference.

    Parses command-line arguments to determine the source of image paths (either a directory or a list file),
    initializes the prediction model with the specified parameters, generates substrate predictions for the images,
    and prints the head of the resulting DataFrame.

    Steps:
        1. Parse command-line arguments.
        2. Read image paths from a directory or a list file.
        3. Initialize the predictor with provided classes, batch, name, image size, and weights directory.
        4. Generate substrate predictions.
        5. Print the first few rows of the prediction results.

    Returns:
        None
    """
    print_warranty()
    args = parse_args()
    if args.image_dir:
        img_pth_lst = Utils.read_image_path_to_lst(args.image_dir)
    elif args.image_list_file:
        img_pth_lst = Utils.read_image_lst_txt(args.image_list_file)
    else:
        print("No valid image source provided.")
        return  # Stop execution if no valid image source is found
    predictor = Predict.from_class_type(
        img_pth_lst=img_pth_lst,
        classes=args.classes,
        start_batch=args.start_batch,
        name=args.name,
        img_size=(1306, 2458),
        weights_dir=args.weights_dir
    )
    df = predictor.generate_substrate_pred()
    print(df.head())

if __name__ == "__main__":
    main()