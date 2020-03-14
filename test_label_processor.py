import argparse
import os
import cv2
import numpy as np

from tqdm import tqdm
from semantic_segmentation.data_structure.image_handler import ImageHandler
from semantic_segmentation.preprocessing.label_preprocessor import LabelPreProcessor


def main(args_):
    df = args_.dataset_folder
    lp = LabelPreProcessor("ellipse")

    ldir = os.path.join(df, "labels")
    rdir = os.path.join(df, "new_lab")
    if not os.path.isdir(rdir):
        os.mkdir(rdir)
    for l in tqdm(os.listdir(ldir)):
        if l.endswith(".png"):
            lbm = cv2.imread(os.path.join(ldir, l))

            lbm = lp.apply(lbm)
            new_lbm = ImageHandler(lbm)

            cv2.imwrite(os.path.join(rdir, l), new_lbm.normalize())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    parser.add_argument(
        "--model_folder", "-m", default="./test", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
