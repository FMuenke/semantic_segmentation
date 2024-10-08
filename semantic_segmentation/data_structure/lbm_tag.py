import os
import cv2
import numpy as np

from semantic_segmentation.data_structure.image_handler import ImageHandler


class LbmTag:
    def __init__(self, path_to_image_file, color_coding):
        self.path_to_image_file = path_to_image_file
        self.path_to_label_file = self.get_pot_label_path()

        self.color_coding = color_coding

        if self.path_to_label_file.endswith(".npy"):
            self.full_label_map = True
        else:
            self.full_label_map = False

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
            ".png", ".jpg", ".jpeg", ".tif", ".tiff", "_label.tiff", "_label.tif", "_label.png",
            "_segmentation.png", "GT.png", ".npy", "_label_ground-truth.png", "_lab.png"
        ]

        for ext in extensions:
            pot_label_name = os.path.join(base_dir, base_name + ext)
            if os.path.isfile(pot_label_name):
                return pot_label_name
        return "None"

    def load_x(self):
        img = cv2.imread(self.path_to_image_file)
        if img is None:
            print(self.path_to_image_file)
        return img

    def load_y_as_color_map(self, label_size):
        y_img = np.zeros((label_size[0], label_size[1], 3))
        if self.path_to_label_file == "None":
            return y_img

        if self.full_label_map:
            lbm = np.load(self.path_to_label_file)
            lbm = cv2.resize(lbm, (label_size[1], label_size[0]), interpolation=cv2.INTER_NEAREST)
            for idx, cls in enumerate(self.color_coding):
                iy, ix = np.where(lbm[:, :, idx] == 1)
                y_img[iy, ix, :] = [self.color_coding[cls][1][0],
                                    self.color_coding[cls][1][1],
                                    self.color_coding[cls][1][2]]
            return y_img
        else:
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
                y_img[iy, ix, :] = [self.color_coding[cls][1][0],
                                    self.color_coding[cls][1][1],
                                    self.color_coding[cls][1][2]]
            return y_img

    def load_y(self, label_size, label_prep=None):
        y_img = np.zeros((label_size[0], label_size[1], len(self.color_coding)))
        if self.path_to_label_file == "None":
            return y_img
        if self.full_label_map:
            y_img = np.load(self.path_to_label_file)
            classes_to_sample = [self.color_coding[cls][0] for cls in self.color_coding]
            y_img = y_img[:, :, classes_to_sample]
            return y_img
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

            y_img_cls = np.zeros((label_size[0], label_size[1]))
            y_img_cls[c == 3] = 1
            y_img[:, :, idx] = y_img_cls

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
        # Classes are compared by comparing the three color values in the image for every pixel
        height, width = color_map.shape[:2]
        if self.path_to_label_file != "None":
            lbm = self.load_y_as_color_map((height, width))
            for idx, cls in enumerate(self.color_coding):
                cls_key = self.color_coding[cls][1]
                c00 = np.zeros((height, width))
                c01 = np.zeros((height, width))
                c02 = np.zeros((height, width))
                c00[lbm[:, :, 0] == cls_key[0]] = 1
                c01[lbm[:, :, 1] == cls_key[1]] = 1
                c02[lbm[:, :, 2] == cls_key[2]] = 1

                c10 = np.zeros((height, width))
                c11 = np.zeros((height, width))
                c12 = np.zeros((height, width))
                c10[color_map[:, :, 0] == cls_key[0]] = 1
                c11[color_map[:, :, 1] == cls_key[1]] = 1
                c12[color_map[:, :, 2] == cls_key[2]] = 1

                c0 = c00 + c01 + c02
                c1 = c10 + c11 + c12

                tp_map = np.zeros((height, width))
                fp_map = np.zeros((height, width))
                fn_map = np.zeros((height, width))

                tp_map[np.logical_and(c0 == 3, c1 == 3)] = 1
                fp_map[np.logical_and(c0 != 3, c1 == 3)] = 1
                fn_map[np.logical_and(c0 == 3, c1 != 3)] = 1
                stats_handler.count(cls, "tp", np.sum(tp_map))
                stats_handler.count(cls, "fp", np.sum(fp_map))
                stats_handler.count(cls, "fn", np.sum(fn_map))

    def visualize_result(self, vis_path, color_map):
        im_id = os.path.basename(self.path_to_image_file)
        vis_file = os.path.join(vis_path, im_id)
        img_h = ImageHandler(self.load_x())
        cv2.imwrite(vis_file, img_h.overlay(color_map))
