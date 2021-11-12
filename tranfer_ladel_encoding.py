import argparse
import os
import cv2


def main(args_):
    mapping = {
        "crack": [[255, 255, 255], [1, 1, 1]]
    }


    old_labels = args_.old_labels
    new_labels = args_.new_labels


    for f in os.listdir(old_labels):
        lbm = cv2.imread(os.path.join(old_labels, f))




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_labels",
        "-old",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    parser.add_argument(
        "--new_labels",
        "-new",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
