import os
import json
from semantic_segmentation.data_structure.folder import Folder


class Config:
    def __init__(self):

        self.color_coding = {
            # "crack": [[1, 1, 1], [0, 0, 255]],
            # "patch": [[2, 2, 2], [0, 100, 0]],
            # "outburst": [[4, 4, 4], [0, 100, 100]],
            # "filled_crack": [[6, 6, 6], [100, 100, 0]],
            # "manhole": [[9, 9, 9], [200, 0, 0]],
            "ellipse": [[255, 255, 255], [0, 0, 255]],
            # "top": [[50, 50, 50], [255, 0, 0]],
            # "edge": [[100, 100, 100], [0, 255, 0]],
            # "edge_lowered": [[150, 150, 150], [0, 0, 255]],
            # "bg": [[0, 0, 0], [0, 0, 0]],
        }

        self.opt = {
            "backbone": "unet",
            "logistic": "sigmoid",
            "loss": "dice",
            "label_prep": "basic",
            "optimizer": "adam",
            "input_shape": [256, 256, 3],
            "batch_size": 8,
            "use_augmentation": True,
            "init_learning_rate": 10e-5,
        }

        self.randomized_split = False


def load_config(model_dir):
    print("Load cfg from model directory")
    color_coding_path = os.path.join(model_dir, "color_coding.json")
    opt_path = os.path.join(model_dir, "opt.json")
    cfg = Config()
    cfg.color_coding = load_dict(color_coding_path)
    cfg.opt = load_dict(opt_path)
    return cfg


def save_config(model_dir, cfg):
    print("config.pickle is saved to {}".format(model_dir))
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
