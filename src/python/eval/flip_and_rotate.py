
import cv2
import argparse
import glob
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flips and rotate image ."
    )
    parser.add_argument(
        "--dataset_path_1",
        type=str,
        help="Path to images from first dataset - the postcontrast base images",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"args for flip and rotate images computation: {args}")
    types = (args.dataset_path_1 +'/*.png', args.dataset_path_1 + '/*.jpg')
    files_grabbed = []
    for filetype in types:
        files_grabbed.extend(glob.glob(filetype))
    for image in tqdm(files_grabbed):
        img = cv2.imread(image)
        img = cv2.rotate(img, rotateCode = 0) # 90 degrees clockwise
        img = cv2.flip(img, 1) # horizontal flip
        cv2.imwrite(image, img)

