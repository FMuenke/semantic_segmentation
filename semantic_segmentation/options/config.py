import os
import pickle
from semantic_segmentation.data_structure.folder import Folder


class Config:
    def __init__(self):

        self.color_coding = {
            "man_hole": [[1, 1, 1], [255, 0, 0]]
        }

        self.backbone = "unet"
        self.input_shape = [512, 512, 3]
        self.batch_size = 4

        self.use_augmentation = True

        self.class_mapping = dict()


def load_config(model_dir):
    print("Load cfg from model directory")
    cfg_path = os.path.join(model_dir, "config.pickle")
    with open(cfg_path, "rb") as f_in:
        cfg = pickle.load(f_in)
    return cfg


def save_config(model_dir, cfg):
    print("config.pickle is saved to {}".format(model_dir))
    fol = Folder(model_dir)
    fol.check_n_make_dir()
    with open(os.path.join(model_dir, "config.pickle"), "wb") as config_f:
        pickle.dump(cfg, config_f)
