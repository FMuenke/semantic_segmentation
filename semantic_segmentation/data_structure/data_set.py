import os
from tqdm import tqdm
import numpy as np
from semantic_segmentation.data_structure.folder import Folder

from semantic_segmentation.data_structure.lbm_tag import LbmTag


class DataSet:
    def __init__(self, path_to_data_set, color_coding):
        self.path_to_data_set = path_to_data_set

        self.color_coding = color_coding

    def _load(self, tag_set, summary, path):
        path = Folder(path)
        img_path = Folder(os.path.join(str(path), "images"))
        print("Try Loading Data from: {}".format(path))
        if img_path.exists():
            print("loading...")
            for img_f in tqdm(sorted(os.listdir(str(img_path)))):
                if img_f.endswith((".jpg", ".png", "tif")):
                    tag_set[len(tag_set)] = LbmTag(os.path.join(str(img_path), img_f),
                                                   self.color_coding)
                    unique, counts = tag_set[len(tag_set)-1].summary()
                    for u, c in zip(unique, counts):
                        if u not in summary:
                            summary[u] = c
                        else:
                            summary[u] += c
        else:
            for f in os.listdir(str(path)):
                file_name = os.path.join(str(path), f)
                if os.path.isdir(file_name):
                    self._load(tag_set, summary, file_name)

    def load(self):
        tag_set = dict()
        summary = dict()
        self._load(tag_set, summary, self.path_to_data_set)

        print("DataSet Summary:")
        tot = 0
        for u in summary:
            tot += summary[u]
        for u in summary:
            print("ClassIdx {}: {}".format(u, summary[u]/tot))

        assert len(tag_set) > 0, "No Data was found.."
        return tag_set

    def split(self, tag_set, percentage=0.2, random=True):
        train_set = []
        validation_set = []
        if random:
            dist = np.random.permutation(len(tag_set))
        else:
            dist = range(len(tag_set))
        for d in dist:
            if len(validation_set) > percentage * len(tag_set):
                train_set.append(tag_set[d])
            else:
                validation_set.append(tag_set[d])
        print("Training Samples: {}".format(len(train_set)))
        print("Validation Samples: {}".format(len(validation_set)))
        print(" ")
        return np.array(train_set), np.array(validation_set)
