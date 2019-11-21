import argparse

from convolutional_neural_network.model_handler import ModelHandler

from data_structure.data_set import DataSet

from options.config import Config, save_config


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder

    cfg = Config()

    model_handler = ModelHandler(mf, cfg)
    model_handler.build()

    save_config(model_dir=mf, cfg=cfg)

    d_set = DataSet(df, cfg.color_coding)
    tag_set = d_set.load()
    train_set, validation_set = d_set.split(tag_set)

    model_handler.fit(train_set, validation_set)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with predictions",
    )
    parser.add_argument(
        "--model_folder", "-m", default="./test", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
