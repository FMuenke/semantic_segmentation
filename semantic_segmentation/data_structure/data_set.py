import os
import numpy as np
from semantic_segmentation.data_structure.folder import Folder

from semantic_segmentation.data_structure.lbm_tag import LbmTag


class DataSet:
    def __init__(self, path_to_data_set, color_coding):
        self.path_to_data_set = path_to_data_set
        self.img_folder = Folder(os.path.join(self.path_to_data_set, "images"))
        self.lbm_folder = Folder(os.path.join(self.path_to_data_set, "labels"))

        self.color_coding = color_coding

    def load(self):
        assert self.img_folder.exists() and self.lbm_folder.exists(), "Abort, No data set to load found..."

        tag_set = dict()
        summary = {}
        for img_f in os.listdir(self.img_folder.path()):
            if img_f.endswith((".jpg", ".png", "tif")):
                tag_set[len(tag_set)] = LbmTag(os.path.join(self.img_folder.path(), img_f),
                                               self.color_coding)
                unique, counts = tag_set[len(tag_set)-1].summary()
                for u, c in zip(unique, counts):
                    if u not in summary:
                        summary[u] = c
                    else:
                        summary[u] += c
        print("DataSet Summary:")
        tot = 0
        for u in summary:
            tot += summary[u]
        for u in summary:
            print("ClassIdx {}: {}".format(u, summary[u]/tot))
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
