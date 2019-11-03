import os
import numpy as np


class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.obj_folder = os.path.join(self.data_folder, "labels")

        self.obj_ids = self.get_obj_ids()

        print("Number of Ids: {}".format(len(self.obj_ids)))

    def get_obj_ids(self):
        obj_ids = []
        for f in os.listdir(self.obj_folder):
            if not f[0] == ".":
                if "augmented" not in f:
                    obj_ids.append(f[:-4])
        obj_ids = np.array(obj_ids)
        return obj_ids

    def split_train_validation(self, ratio=0.2, mode="naive"):
        size_train = int(len(self.obj_ids)*(1-ratio))
        size_validation = int(len(self.obj_ids)*ratio)
        training_idx = np.random.randint(self.obj_ids.shape[0], size=size_train)
        validation_idx = np.random.randint(self.obj_ids.shape[0], size=size_validation)
        print("Training Samples: {}".format(len(training_idx)))
        print("Validation Samples: {}".format(len(validation_idx)))
        print(" ")
        return self.obj_ids[training_idx], self.obj_ids[validation_idx]