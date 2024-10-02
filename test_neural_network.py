import argparse
import os

from tqdm import tqdm

from semantic_segmentation.options.config import load_config
from semantic_segmentation.data_structure.data_set import DataSet
from semantic_segmentation.data_structure.folder import Folder
from semantic_segmentation.convolutional_neural_network.semantic_segmentation_model import SemanticSegmentationModel

from semantic_segmentation.data_structure.stats_handler import StatsHandler


def main(args_):
    df = args_.dataset_folder
    mf = args_.model_folder
    conf = float(args_.confidence)
    cfg = load_config(model_dir=mf)
    model_handler = SemanticSegmentationModel(mf, cfg)
    model_handler.batch_size = 1
    model_handler.build()

    res_fol = Folder(os.path.join(mf, "segmentations"))
    res_fol.check_n_make_dir(clean=True)

    vis_fol = Folder(os.path.join(mf, "overlays"))
    vis_fol.check_n_make_dir(clean=True)

    d_set = DataSet(df, cfg.color_coding)
    t_set = d_set.load()

    sh = StatsHandler(cfg.color_coding)
    for tid in tqdm(t_set):
        color_map = model_handler.predict(t_set[tid].load_x(), conf)
        t_set[tid].eval(color_map, sh)
        if args_.plot == "True":
            t_set[tid].write_result(res_fol.path(), color_map)
            t_set[tid].visualize_result(vis_fol.path(), color_map)

    sh.eval()
    sh.show()
    if conf == 0.5:
        sh.write_report(os.path.join(mf, "report.txt"))
    else:
        sh.write_report(os.path.join(mf, "report_{}.txt".format(conf)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_folder",
        "-df",
        default="./data/train",
        help="Path to directory with images and labels folder",
    )
    parser.add_argument(
        "--model_folder", "-model", default="./test", help="Path to model directory"
    )
    parser.add_argument("--confidence", "-conf", default=0.5, help="Confidence to accept a classification")
    parser.add_argument("--plot", "-p", default="False", help="Plot the results [TRUE/FALSE]")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
