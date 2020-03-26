import argparse
import os

from tqdm import tqdm

from semantic_segmentation.options.config import load_config
from semantic_segmentation.data_structure.data_set import DataSet
from semantic_segmentation.data_structure.folder import Folder
from semantic_segmentation.convolutional_neural_network.model_handler import ModelHandler


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    cfg = load_config(model_dir=mf)
    model_handler = ModelHandler(mf, cfg)
    model_handler.batch_size = 1
    model_handler.build()

    res_fol = Folder(os.path.join(df, "inference"))
    res_fol.check_n_make_dir(clean=True)

    d_set = DataSet(df, cfg.color_coding)
    t_set = d_set.load()

    for tid in tqdm(t_set):
        color_map = model_handler.predict(t_set[tid].load_x())
        t_set[tid].write_inference(str(res_fol), color_map)


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
