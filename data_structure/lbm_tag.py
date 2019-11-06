import os
import cv2
import numpy as np


class LbmTag:
    def __init__(self, path_to_image_file, color_coding):
        self.path_to_image_file = path_to_image_file
        self.path_to_label_file = self.get_pot_label_path()

        self.color_coding = color_coding

    def get_pot_label_path(self):
        """
        Used to guess the matching ground truth labelfile
        Args:
            img_id: complete path to image

        Returns:
            estimated full path to label file
        """
        path_to_label_file = self.path_to_image_file.replace("/images/", "/labels/")

        path_to_label_file = path_to_label_file[:-4] + ".png"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file
        path_to_label_file = path_to_label_file[:-4] + ".tif"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file
        path_to_label_file = path_to_label_file[:-4] + ".tiff"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file
        path_to_label_file = path_to_label_file[:-4] + "_label.tiff"
        if os.path.isfile(path_to_label_file):
            return path_to_label_file

        return None

    def load_x(self):
        return cv2.imread(self.path_to_label_file)

    def load_y(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1], len(self.color_coding)))
        if self.path_to_label_file is not None:
            lbm = cv2.imread(self.path_to_label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                for x in range(label_size[1]):
                    for y in range(label_size[0]):
                        if lbm[y, x, 0] == self.color_coding[cls][0][2] \
                                and lbm[y, x, 1] == self.color_coding[cls][0][1] \
                                and lbm[y, x, 2] == self.color_coding[cls][0][0]:
                            y_img[y, x, idx] = 1

        return cv2.resize(y_img, (300, 300), interpolation=cv2.INTER_NEAREST)
