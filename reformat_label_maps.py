import argparse
import os
import cv2

from tqdm import tqdm


def main(args_):
    instructions = {
        2: 3
    }
    for label_file in tqdm(os.listdir(args_.label_folder)):
        if label_file.endswith(".tif"):
            lbf = os.path.join(args_.label_folder, label_file)
            lbm = cv2.imread(lbf)

            height, width = lbm.shape[:2]
            for i in range(width):
                for j in range(height):
                    for t in instructions:
                        if t == lbm[j, i, 0]:
                            lbm[j, i, :] = instructions[t]
            cv2.imwrite(lbf, lbm)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_folder",
        "-lf",
        default="./data/train",
        help="Path to directory with the original labels",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)