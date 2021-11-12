import os
import cv2
import numpy as np

from semantic_segmentation.data_structure.image_handler import ImageHandler
from semantic_segmentation.preprocessing.label_preprocessor import LabelPreProcessor


class LbmTag:
    def __init__(self, path_to_image_file, color_coding):
        self.path_to_image_file = path_to_image_file
        self.path_to_label_file = self.get_pot_label_path()

        self.color_coding = color_coding

    def summary(self):
        y = self.load_y([100, 100])
        unique = [0]
        counts = [100*100]
        for i in range(y.shape[2]):
            u, c = np.unique(y[:, :, i], return_counts=True)
            unique.append(i + 1)
            if len(c) > 1:
                counts.append(c[1])
            else:
                counts.append(0)
        return unique, counts

    def get_pot_label_path(self):
        """
        Used to guess the matching ground truth labelfile
        Args:
            img_id: complete path to image

        Returns:
            estimated full path to label file
        """
        base_name = os.path.basename(self.path_to_image_file)[:-4]
        base_dir = os.path.dirname(self.path_to_image_file).replace("images", "labels")

        extensions = [
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", "_label.tiff", "_label.tif", "GT.png"
        ]

        for ext in extensions:
            pot_label_name = os.path.join(base_dir, base_name + ext)
            if os.path.isfile(pot_label_name):
                return pot_label_name
        return None

    def load_x(self):
        img = cv2.imread(self.path_to_image_file)
        if img is None:
            print(self.path_to_image_file)
        return img

    def load_y_as_color_map(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1], 3))
        if self.path_to_label_file is not None:
            lbm = cv2.imread(self.path_to_label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                c0 = np.zeros((label_size[0], label_size[1]))
                c1 = np.zeros((label_size[0], label_size[1]))
                c2 = np.zeros((label_size[0], label_size[1]))

                c0[lbm[:, :, 0] == self.color_coding[cls][0][2]] = 1
                c1[lbm[:, :, 1] == self.color_coding[cls][0][1]] = 1
                c2[lbm[:, :, 2] == self.color_coding[cls][0][0]] = 1
                c = c0 + c1 + c2
                iy, ix = np.where(c == 3)
                y_img[iy, ix, :] = [self.color_coding[cls][1][2],
                                    self.color_coding[cls][1][1],
                                    self.color_coding[cls][1][0]]
        return y_img

    def load_y(self, label_size, label_prep=None):
        y_img = np.zeros((label_size[0], label_size[1], len(self.color_coding)))
        if self.path_to_label_file is not None:
            lbm = cv2.imread(self.path_to_label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                c0 = np.zeros((label_size[0], label_size[1]))
                c1 = np.zeros((label_size[0], label_size[1]))
                c2 = np.zeros((label_size[0], label_size[1]))

                c0[lbm[:, :, 0] == self.color_coding[cls][0][2]] = 1
                c1[lbm[:, :, 1] == self.color_coding[cls][0][1]] = 1
                c2[lbm[:, :, 2] == self.color_coding[cls][0][0]] = 1
                c = c0 + c1 + c2
                y_img[c == 3] = idx + 1
        lab_pro = LabelPreProcessor(label_prep)
        y_img = lab_pro.apply(y_img)
        return y_img

    def write_result(self, res_path, color_map):
        im_id = os.path.basename(self.path_to_image_file)
        h, w = color_map.shape[:2]
        label = self.load_y_as_color_map((h, w))
        border = 255 * np.ones((h, 10, 3))
        r = np.concatenate([label, border, color_map], axis=1)
        res_file = os.path.join(res_path, im_id[:-4] + ".png")
        cv2.imwrite(res_file, r)

    def write_inference(self, res_path, color_map):
        im_id = os.path.basename(self.path_to_image_file)
        res_file = os.path.join(res_path, im_id[:-4] + ".png")
        cv2.imwrite(res_file, color_map)

    def eval(self, color_map, stats_handler):
        height, width = color_map.shape[:2]
        if self.path_to_label_file is not None:
            lbm = cv2.imread(self.path_to_label_file)
            lbm = cv2.resize(lbm, (width, height), interpolation=cv2.INTER_NEAREST)
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
