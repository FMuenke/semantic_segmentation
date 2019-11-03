import os
import numpy as np
from data_structure.folder import Folder

from data_structure.lbm_tag import LbmTag


class DataSet:
    def __init__(self, path_to_data_set, color_coding):
        self.path_to_data_set = path_to_data_set
        self.img_folder = Folder(os.path.join(self.path_to_data_set, "images"))
        self.lbm_folder = Folder(os.path.join(self.path_to_data_set, "labels"))

        self.color_coding = color_coding

    def load(self):
        assert self.img_folder.exists() and self.lbm_folder.exists(), "Abort, No data set to load found..."

        tag_set = dict()
        for img_f in os.listdir(self.img_folder.path()):
            tag_set[len(tag_set)] = LbmTag(os.path.join(self.img_folder.path(), img_f),
                                           self.color_coding)
        return tag_set

    def split(self, tag_set, percentage=0.2):
        train_set = []
        validation_set = []
        dist = np.random.permutation(len(tag_set))
        for d in dist:
            if len(validation_set) > percentage * len(tag_set):
                train_set.append(tag_set[d])
            else:
                validation_set.append(tag_set[d])
        print("Training Samples: {}".format(len(train_set)))
        print("Validation Samples: {}".format(len(validation_set)))
        print(" ")
        return np.array(train_set), np.array(validation_set)
