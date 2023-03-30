import os
import json
import tensorflow as tf
from semantic_segmentation.data_structure.folder import Folder


class Config:
    def __init__(self):

        self.color_coding = {
            "top": [[255, 255, 255], [255, 255, 255]],
            # "edge": [1, [10, 217, 252]],
            # "edge_lowered": [2, [150, 101, 146]],
            # "curbstone": [3, [22, 124, 99]],
            # "leaves": [4, [232, 139, 150]],
        }

        self.opt = {
            "backbone": "unet-resnet18-imagenet",
            "logistic": "sigmoid",
            "loss": "bc",
            "label_prep": "basic",
            "optimizer": "adam",
            "input_shape": [256, 256, 3],
            "batch_size": 8,
            "use_augmentation": True,
            "init_learning_rate": 10e-4,  # 0.00001 (deeplab)
            "tf-version": tf.__version__,
        }

        self.randomized_split = False


def load_config(model_dir):
    print("[INFO] Load cfg from model directory")
    color_coding_path = os.path.join(model_dir, "color_coding.json")
    opt_path = os.path.join(model_dir, "opt.json")
    cfg = Config()
    cfg.color_coding = load_dict(color_coding_path)
    cfg.opt = load_dict(opt_path)
    return cfg


def save_config(model_dir, cfg):
    print("[INFO] config.pickle is saved to {}".format(model_dir))
    fol = Folder(model_dir)
    fol.check_n_make_dir()
    color_coding_path = os.path.join(model_dir, "color_coding.json")
    opt_path = os.path.join(model_dir, "opt.json")
    save_dict(cfg.color_coding, color_coding_path)
    save_dict(cfg.opt, opt_path)


def save_dict(dict_to_save, path_to_save):
    with open(path_to_save, "w") as f:
        j_file = json.dumps(dict_to_save)
        f.write(j_file)


def load_dict(path_to_load):
    with open(path_to_load) as json_file:
        dict_to_load = json.load(json_file)
    return dict_to_load
