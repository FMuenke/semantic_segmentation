import argparse
import os
from shutil import copyfile


def main(args):
    if not os.path.isdir(args.formatted_folder):
        os.mkdir(args.formatted_folder)
    for label_file in os.listdir(args.label_folder):
        if "_color" in label_file:
            continue
        if "_label" in label_file:
            new_file_name = os.path.join(args.formatted_folder, label_file.replace("_label", ""))
            old_file_name = os.path.join(args.label_folder, label_file)
            copyfile(old_file_name, new_file_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_folder",
        "-lf",
        default="./data/train",
        help="Path to directory with the original labels",
    )
    parser.add_argument(
        "--formatted_folder",
        "-ff",
        default="./data/train",
        help="Path to directory where formatted labels are saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
