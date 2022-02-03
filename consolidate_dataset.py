import argparse
import os
import cv2
import numpy as np

from semantic_segmentation.data_structure.segmentation_data import SegmentationData


def check_n_make(path):
    if not os.path.isdir(path):
        os.mkdir(path)


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
    old_data = args_.old_data
    new_data = args_.new_data

    check_n_make(new_data)
    check_n_make(os.path.join(new_data, "images"))
    check_n_make(os.path.join(new_data, "labels"))

    for i, img_f in enumerate(os.listdir(os.path.join(old_data, "images"))):
        seg_data = SegmentationData(old_data, img_f[:-4])
        img = cv2.imread(seg_data.image_file)
        lab = cv2.imread(seg_data.label_file)
        if img is None or lab is None:
            continue
        cv2.imwrite(os.path.join(new_data, "images", str(i) + ".png"), img)
        cv2.imwrite(os.path.join(new_data, "labels", str(i) + ".png"), lab)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old_data",
        "-old",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    parser.add_argument(
        "--new_data",
        "-new",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
