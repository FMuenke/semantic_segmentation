import argparse
import os
import cv2
import numpy as np


def convert_lbm(lbm, mapping):
    label_size = lbm.shape
    new_lbm = np.zeros(label_size)
    for idx, cls in enumerate(mapping):
        for x in range(label_size[1]):
            for y in range(label_size[0]):
                if lbm[y, x, 0] == mapping[cls][0][2] \
                        and lbm[y, x, 1] == mapping[cls][0][1] \
                        and lbm[y, x, 2] == mapping[cls][0][0]:
                    new_lbm[y, x, :] = mapping[cls][1]
    return new_lbm


def main(args_):
    mapping = {
        "crack": [[255, 255, 255], [1, 1, 1]]
    }

    old_labels = args_.old_labels
    new_labels = args_.new_labels

    if not os.path.isdir(new_labels):
        os.mkdir(new_labels)

    for f in os.listdir(old_labels):
        lbm = cv2.imread(os.path.join(old_labels, f))
        lbm = convert_lbm(lbm, mapping)
        cv2.imwrite(os.path.join(new_labels, f), lbm)




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
