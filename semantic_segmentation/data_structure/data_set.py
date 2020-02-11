import os
import numpy as np
from semantic_segmentation.data_structure.folder import Folder

from semantic_segmentation.data_structure.lbm_tag import LbmTag


class DataSet:
    def __init__(self, path_to_data_set, color_coding):
        self.path_to_data_set = path_to_data_set
        self.img_folder = Folder(os.path.join(self.path_to_data_set, "images"))

        self.color_coding = color_coding

    def _load(self, tag_set, summary, path):
        img_path = Folder(os.path.join(path, "images"))
        if img_path.exists():
            for img_f in os.listdir(str(img_path)):
                file_name = os.path.join(str(img_path), img_f)
                if os.path.isdir(file_name):
                    self._load(tag_set, summary, file_name)
                if img_f.endswith((".jpg", ".png", "tif")):
                    tag_set[len(tag_set)] = LbmTag(os.path.join(str(img_path), img_f),
                                                   self.color_coding)
                    unique, counts = tag_set[len(tag_set)-1].summary()
                    for u, c in zip(unique, counts):
                        if u not in summary:
                            summary[u] = c
                        else:
                            summary[u] += c

    def load(self):
        assert self.img_folder.exists(), "Abort, No data set to load found..."

        tag_set = dict()
        summary = dict()
        self._load(tag_set, summary, str(self.img_folder))

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
