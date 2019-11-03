import os
import pickle


class Config:
    def __init__(self):

        self.color_coding = {
            "crack": [[255, 255, 255], [255, 0, 0]]
        }

        self.backbone = "resnet"
        self.input_size = None

        self.use_augmentation = True

        self.class_mapping = dict()
        self.compile()

    def compile(self):
        assert 'bg' not in self.color_coding,\
            "class bg is added automatically and should not be manually included!"

        self.class_mapping["bg"] = len(self.class_mapping)
        for class_name in self.color_coding:
            self.class_mapping[class_name] = len(self.class_mapping)


def load_config(model_dir):
    print("Load cfg from model directory")
    cfg_path = os.path.join(model_dir, "config.pickle")
    with open(cfg_path, "rb") as f_in:
        cfg = pickle.load(f_in)
    return cfg


def save_config(model_dir, cfg):
    print("config.pickle is saved to {}".format(model_dir))
    with open(os.path.join(model_dir, "config.pickle"), "wb") as config_f:
        pickle.dump(cfg, config_f)
