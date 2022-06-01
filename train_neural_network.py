import argparse
import numpy as np

from semantic_segmentation.convolutional_neural_network.semantic_segmentation_model import SemanticSegmentationModel

from semantic_segmentation.data_structure.data_set import DataSet

from semantic_segmentation.options.config import Config, save_config


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder

    cfg = Config()

    model_handler = SemanticSegmentationModel(mf, cfg)
    model_handler.build()

    save_config(model_dir=mf, cfg=cfg)

    d_set = DataSet(df, cfg.color_coding)
    tag_set = d_set.load()

    number_to_train_on = int(args_.number_training_images)
    if number_to_train_on != 0:
        tags = [tag_set[t] for t in tag_set]
        seed_id = int(mf.split("-RUN-")[-1])
        rng = np.random.default_rng(seed_id)
        train_set = rng.choice(tags, number_to_train_on, replace=False)
        number_to_train_on = np.min([number_to_train_on, len(tags) - 1])
        validation_set = rng.choice(tags, number_to_train_on, replace=False)
        print("Number of Training images reduced! - {}/{} -".format(len(train_set), len(validation_set)))
        model_handler.fit(train_set, validation_set)
    else:
        train_set, validation_set = d_set.split(tag_set, percentage=0.5, random=cfg.randomized_split)
        model_handler.fit(train_set, validation_set)


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
