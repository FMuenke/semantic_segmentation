import os
import cv2
import numpy as np

from semantic_segmentation.data_structure.image_handler import ImageHandler


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

        return cv2.imread(self.path_to_image_file)

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
        return y_img

    def write_result(self, res_path, color_map):
        im_id = os.path.basename(self.path_to_image_file)
        res_file = os.path.join(res_path, im_id)
        cv2.imwrite(res_file, color_map)

    def eval(self, color_map, stats_handler):
        height, width = color_map.shape[:2]
        if self.path_to_label_file is not None:
            lbm = cv2.imread(self.path_to_label_file)
            lbm = cv2.resize(lbm, (height, width), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                for x in range(width):
                    for y in range(height):
                        a = lbm[y, x, :] == self.color_coding[cls][0]
                        b = color_map[y, x, :] == self.color_coding[cls][1]
                        if a.all():
                            if b.all():
                                stats_handler.count(cls, "tp")
                            else:
                                stats_handler.count(cls, "fn")
                        else:
                            if b.all():
                                stats_handler.count(cls, "fp")

    def visualize_result(self, vis_path, color_map):
        im_id = os.path.basename(self.path_to_image_file)
        vis_file = os.path.join(vis_path, im_id)
        img_h = ImageHandler(self.load_x())
        cv2.imwrite(vis_file, img_h.overlay(color_map))