import argparse
import os

import matplotlib.pyplot as plt

from semantic_segmentation.data_structure.data_set import DataSet

import numpy as np
from cleanlab.segmentation.filter import find_label_issues
from cleanlab.segmentation.rank import get_label_quality_scores, issues_from_scores
from cleanlab.segmentation.summary import display_issues, common_label_issues, filter_by_class
np.set_printoptions(suppress=True)


def main(args_):
    df = args_.dataset_folder

    res_fol = os.path.join(df, "inference")
    color_coding = {
        "top": [[255, 255, 255], [255, 255, 255]],
    }

    classes = [
        'background',
        'filled_crack'
    ]

    d_set = DataSet(df, color_coding)
    t_set = d_set.load()

    labels = []
    pred_probs = []

    image_mapping = {}

    for i, tid in enumerate(t_set):
        im_id = os.path.basename(t_set[tid].path_to_image_file)

        y_proba = np.load(os.path.join(res_fol, im_id[:-4] + ".npy"))
        stamp = np.ones((256, 256, 1))
        stamp = stamp - y_proba
        y_proba = np.concatenate([stamp, y_proba], axis=2)
        y_label = t_set[tid].load_y([256, 256])
        image_mapping[i] = im_id

        y_label = np.concatenate([0.5 * np.ones((256, 256, 1)), y_label], axis=2)
        labels.append(np.argmax(y_label, axis=2))
        pred_probs.append(np.transpose(y_proba, (2, 0, 1)))

    labels = np.array(labels).astype(int)
    pred_probs = np.array(pred_probs)
    print(labels.shape, pred_probs.shape)

    image_scores, pixel_scores = get_label_quality_scores(labels, pred_probs, n_jobs=None, batch_size=100000)
    list_of_issues = issues_from_scores(image_scores)
    print(list_of_issues)

    issues = find_label_issues(labels, pred_probs, downsample=1, n_jobs=None, batch_size=100000)
    display_issues(issues, labels=labels, pred_probs=pred_probs, class_names=classes, top=10)
    common_label_issues(issues, labels=labels, pred_probs=pred_probs, class_names=classes)


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
