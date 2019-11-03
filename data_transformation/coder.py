import numpy as np
import os
import cv2
from utils.img_preprocessing import pre_process_image
from utils.src_file_utils import check_n_make_dir

from utils.data_augmentation import augment_img_and_label_map


class Coder:
    def __init__(self, data_folder: str, cfg, name="aggregator", use_augmentation=False):
        self.data_folder = data_folder
        self.result_folder = os.path.join(data_folder, 'prediction_by_{}'.format(name))

        if name is not None:
            check_n_make_dir(self.result_folder, clean=True)

        self.mode = cfg.mode
        self.clmp = cfg.clmp
        self.inv_clmp = {v: k for k, v in self.clmp.items()}
        self.target_size = cfg.input_size

        self.norm = cfg.norm
        self.scale = cfg.scale

        self.class_encoding = self.load_class_encoding(data_folder)
        self.image_dir = os.path.join(data_folder, "images")

    def get_pot_label_path(self, img_id):
        """
        Used to guess the matching ground truth labelfile
        Args:
            img_id: complete path to image

        Returns:
            estimated full path to label file
        """
        label_name = img_id + ".png"
        path_to_label_file = os.path.join(self.data_folder, 'labels', label_name)
        if not os.path.isfile(path_to_label_file):
            label_name = img_id + ".jpg"
            path_to_label_file = os.path.join(self.data_folder, 'labels', label_name)
        if not os.path.isfile(path_to_label_file):
            label_name = img_id + ".JPG"
            path_to_label_file = os.path.join(self.data_folder, 'labels', label_name)
        if not os.path.isfile(path_to_label_file):
            label_name = img_id + ".tif"
            path_to_label_file = os.path.join(self.data_folder, 'labels', label_name)
        if not os.path.isfile(path_to_label_file):
            label_name = img_id + ".tiff"
            path_to_label_file = os.path.join(self.data_folder, 'labels', label_name)
        return path_to_label_file

    def get_pot_image_path(self, img_id):
        if os.path.isfile(os.path.join(self.image_dir, img_id + ".jpg")):
            img_file_path = os.path.join(self.image_dir, img_id + ".jpg")
        elif os.path.isfile(os.path.join(self.image_dir, img_id + ".png")):
            img_file_path = os.path.join(self.image_dir, img_id + ".png")
        elif os.path.isfile(os.path.join(self.image_dir, img_id + ".JPG")):
            img_file_path = os.path.join(self.image_dir, img_id + ".JPG")
        else:
            raise ValueError("Error: Image File {} does not exist!".format(img_id))
        return img_file_path

    def get_X_y(self, img_id, augment=False):
        img_file_path = self.get_pot_image_path(img_id)
        lbm_file_path = self.get_pot_label_path(img_id)
        img = cv2.imread(img_file_path)
        if lbm_file_path is not None:
            lbm = cv2.imread(lbm_file_path)
        else:
            lbm = np.zeros(img.shape)

        if augment:
            img, lbm = augment_img_and_label_map(img, lbm)

        X = pre_process_image(img,
                              height=self.target_size[0],
                              width=self.target_size[1],
                              norm=self.norm,
                              scale=self.scale)

        y = self.encode_lbm_output(lbm)

        return X, y

    def write_output(self, img_ids, y_pred, visualize=True):
        if self.mode == "lbm":
            batch_classes = self.decode_lbm_output(y_pred)
            self.write_lbm_prediction(self.result_folder, img_ids, batch_classes)
            if visualize:
                image_path = os.path.join(self.data_folder, "images")
                vis_path = os.path.join(self.data_folder, "visualisation")
                check_n_make_dir(vis_path)
                self.visualize_lbm_prediction(image_path, vis_path, img_ids, batch_classes)
        elif self.mode == "de_shadow":
            batch_classes = []
            for batch_idx in range(y_pred.shape[0]):
                img = 255 * y_pred[batch_idx, :, :, :]
                batch_classes.append(img)
            self.write_lbm_prediction(self.result_folder, img_ids, batch_classes)
        else:
            raise ValueError("Error: Mode not recognised")

    def encode_lbm_output(self, lbm):
        if self.mode == "lbm":
            y_img = np.zeros((self.target_size[0], self.target_size[1], len(self.clmp)))
            lbm = cv2.resize(lbm, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
            for cls in self.clmp:
                if not cls == "bg":
                    if cls in self.class_encoding:
                        for x in range(self.target_size[1]):
                            for y in range(self.target_size[0]):
                                if lbm[y, x, 0] == self.class_encoding[cls][2]\
                                        and lbm[y, x, 1] == self.class_encoding[cls][1]\
                                        and lbm[y, x, 2] == self.class_encoding[cls][0]:
                                    y_img[y, x, self.clmp[cls]] = 1
        else:
            lbm = cv2.resize(lbm, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_CUBIC)
            lbm = lbm[:, :, (2, 1, 0)]
            y_img = lbm / 255
        return y_img

    def load_class_encoding(self, datafolder):
        info_file = os.path.join(datafolder, "info.txt")
        class_encoding = dict()
        if os.path.isfile(info_file):
            with open(info_file) as inf_f:
                for line in inf_f:
                    val1, val2, val3, class_name = line.strip().split(",")
                    class_encoding[class_name] = [int(val1), int(val2), int(val3)]
        else:
            print("No Info.txt file found!!")
            enc_str = ""
            for cls in self.clmp:
                color = list(np.random.choice(range(256), size=3))
                class_encoding[cls] = [color[0], color[1], color[2]]
                enc_str += "{},{},{},{}\n".format(color[0], color[1], color[2], cls)
            with open(info_file, "w") as inf_f:
                inf_f.write(enc_str)

        return class_encoding

    def decode_lbm_output(self, y_pred):
        thresh = 0.49
        batch_labelmaps = []
        for idx in range(y_pred.shape[0]):
            y_img = y_pred[idx, :, :, :]
            cls_map = np.argmax(y_img, axis=2)
            cls_conf = np.max(y_img, axis=2)
            height, width, num_cls = y_img.shape
            out = np.zeros((height, width, 3))
            for x in range(width):
                for y in range(height):
                    pixel_class = self.inv_clmp[cls_map[y, x]]
                    pixel_conf = cls_conf[y, x]
                    if pixel_class in self.class_encoding and pixel_conf > thresh:
                        out[y, x, :] = self.class_encoding[pixel_class]

            batch_labelmaps.append(out)
        return batch_labelmaps

    def write_lbm_prediction(self, result_path, img_ids, batch_labelmaps):
        for idx, img_id in enumerate(img_ids):
            prediction_file_path = os.path.join(result_path, img_id+".png")
            cv2.imwrite(prediction_file_path, batch_labelmaps[idx])

    def visualize_lbm_prediction(self, image_path, result_path, img_ids, batch_labelmaps):
        for idx, img_id in enumerate(img_ids):
            full_path_to_image = self.get_pot_image_path(img_id)
            org_img = cv2.imread(full_path_to_image)
            height, width, ch = org_img.shape
            pre_lbm = np.uint8(cv2.resize(batch_labelmaps[idx], (width, height)))
            added_image = cv2.addWeighted(org_img,0.4,pre_lbm,0.1,0, -1)
            prediction_file_path = os.path.join(result_path, img_id+".jpg")
            cv2.imwrite(prediction_file_path, added_image)
