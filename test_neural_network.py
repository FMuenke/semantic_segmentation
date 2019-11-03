import argparse

from data_transformation.coder import Coder

from neural_network.model_handler import ModelHandler

from data_structure.data_loader import DataLoader
from data_structure.generator import Generator

from utils.config_utils import load_config


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    cfg = load_config(model_dir=mf)
    model_handler = ModelHandler(mf, cfg, inference=False)
    model = model_handler.load_model(model_dir=mf)
    coder = Coder(df, cfg, name="convolutional_neural_network")

    data_loader = DataLoader(df)
    img_ids = data_loader.obj_ids

    gen = Generator(coder, mf, cfg)

    gen.predict(model, img_ids)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    parser.add_argument(
        "--model_folder", "-m", default="./model/test", help="Path to model directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
