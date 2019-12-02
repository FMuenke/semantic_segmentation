import argparse
import os

from tqdm import tqdm

from semantic_segmentation.options.config import load_config
from semantic_segmentation.data_structure.data_set import DataSet
from semantic_segmentation.data_structure.folder import Folder
from semantic_segmentation.convolutional_neural_network.mask_handler import MaskHandler

from semantic_segmentation.data_structure.stats_handler import StatsHandler


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    msk_h = MaskHandler(model_folder=mf, target_classes=[args_.target_class])

    msk_h.mask_data_set(df)


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
    parser.add_argument(
        "--target_class", "-tc", default="man_hole", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
