import os
import json
import tensorflow as tf
from semantic_segmentation.data_structure.folder import Folder


class Config:
    def __init__(self):

        self.color_coding = {
            # "crack": [[255, 255, 255], [255, 255, 255]],
            "crack": [[0, 255, 0], [0, 255, 0]],
            # "pothole": [[255, 0, 0], [255, 0, 0]],
        }

        # backbones: resnet18, efficientnetb0, mobilenetv2, seresnext50, resnext50

        self.opt = {
            "backbone": "unet-resnet18-imagenet",
            "logistic": "sigmoid",
            "loss": "bc",
            "label_prep": "basic",
            "optimizer": "adam",
            "input_shape": [256, 256, 3],  # psp: 384
            "batch_size": 8,
            "init_learning_rate": 10e-4,  # 0.00001 (deeplab)
            "aug_horizontal_flip": True,
            "aug_vertical_flip": True,
            "aug_crop": True,
            "aug_rotation": True,
            "aug_tiny_rotation": False,
            "aug_noise": False,
            "aug_brightening": False,
            "aug_blur": False,
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
