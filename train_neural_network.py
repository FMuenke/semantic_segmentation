import argparse
import os
import numpy as np
from time import time

from semantic_segmentation.convolutional_neural_network.semantic_segmentation_model import SemanticSegmentationModel

from semantic_segmentation.data_structure.data_set import DataSet

from semantic_segmentation.options.config import Config, save_config


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder

    cfg = Config()

    model = SemanticSegmentationModel(mf, cfg)
    model.build()

    save_config(model_dir=mf, cfg=cfg)

    d_set = DataSet(df, cfg.color_coding)
    tag_set = d_set.load()

    t0 = time()
    number_to_train_on = int(args_.number_training_images)
    if number_to_train_on != 0:
        tags = [tag_set[t] for t in tag_set]
        seed_id = int(mf.split("-RUN-")[-1])
        rng = np.random.default_rng(seed_id)
        tags = rng.choice(tags, number_to_train_on, replace=False)
        n_val = int(max(1, number_to_train_on * 0.2))
        train_set = tags[:-n_val]
        validation_set = tags[-n_val:]
        print("Number of Training images reduced! - {}/{} -".format(len(train_set), len(validation_set)))
        model.fit(train_set, validation_set)
    else:
        train_set, validation_set = d_set.split(tag_set, percentage=0.1, random=cfg.randomized_split)
        model.fit(train_set, validation_set)
    with open(os.path.join(mf, "time.txt"), "w") as f:
        f.write("[INFO] done in %0.3fs" % (time() - t0))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder",
        "-model",
        help="Path to model directory"
    )
    parser.add_argument(
        "--number_training_images",
        "-n",
        default=0,
        help="Limit the amount of training data [-1: use all data]"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
