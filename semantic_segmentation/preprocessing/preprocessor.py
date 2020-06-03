import numpy as np
import cv2
from semantic_segmentation.data_structure.image_handler import ImageHandler


class Preprocessor:
    def __init__(self, image_size, padding=False):
        self.image_size = image_size
        self.min_height = 16
        self.min_width = 16
        self.max_height = 900
        self.max_width = 900
        self.do_padding = padding

        self.obox = None

    def resize(self, image):
        img_h = ImageHandler(image)
        if None in self.image_size:
            return self.rescale(image, self.max_width, self.max_height, is_lbm=False)
        return img_h.resize(height=self.image_size[0], width=self.image_size[1])

    def normalize(self, image):
        epsilon = 1e-6
        mean_mat = np.mean(image)
        var_mat = np.var(image)
        if var_mat != 0:
            mat_norm = (image - mean_mat) / var_mat
            min_mat = np.min(mat_norm)
            max_mat = np.max(mat_norm)
            mat_norm = (mat_norm - min_mat) / (max_mat - min_mat + epsilon)
        else:
            mat_norm = np.zeros(image.shape)
        return mat_norm

    def rescale(self, data, max_width, max_height, is_lbm=False):
        height, width = data.shape[0], data.shape[1]
        if height >= max_height:
            new_height = max_height
            new_width = int(width * new_height / height)
            if is_lbm:
                data = cv2.resize(data, (int(new_width), int(new_height)), interpolation=cv2.INTER_NEAREST)
            else:
                data = cv2.resize(data, (int(new_width), int(new_height)), interpolation=cv2.INTER_CUBIC)
        height, width = data.shape[0], data.shape[1]
        if width >= max_width:
            new_width = max_width
            new_height = int(height * new_width / width)
            if is_lbm:
                data = cv2.resize(data, (int(new_width), int(new_height)), interpolation=cv2.INTER_NEAREST)
            else:
                data = cv2.resize(data, (int(new_width), int(new_height)), interpolation=cv2.INTER_CUBIC)
        height, width = data.shape[0], data.shape[1]
        return np.reshape(data, (height, width, -1))

    def pad(self, image):
        img_h = ImageHandler(image)
        height, width, ch = image.shape
        if height > width:
            if height >= self.image_size[0]:
                new_height = self.image_size[0]
                new_width = int(width * new_height / height)
                image = img_h.resize(height=new_height, width=new_width)
        else:
            if width >= self.image_size[1]:
                new_width = self.image_size[1]
                new_height = int(height * new_width / width)
                image = img_h.resize(height=new_height, width=new_width)
        ih, iw = image.shape[:2]
        ph, pw = self.image_size[0], self.image_size[1]
        x = np.zeros((ph, pw, ch))
        sy1 = int(ph/2)-int(ih/2)
        sx1 = int(pw/2)-int(iw/2)
        if ch == 1:
            image = np.expand_dims(image, axis=2)
        x[sy1:sy1+ih, sx1:sx1+iw, :] = image
        self.obox = [sx1, sy1, sx1 + iw, sy1 + ih]
        return x

    def apply(self, image):
        if self.do_padding:
            image = self.pad(image)
        else:
            image = self.resize(image)
        img_h = ImageHandler(image)
        if self.image_size[2] == 1:
            image = img_h.gray()
            image = np.expand_dims(image, axis=2)
        image = self.normalize(image)
        return image

    def lbm_resize(self, lbm, width, height):
        if None in [width, height]:
            return self.rescale(lbm, self.max_width, self.max_height, is_lbm=True)
        return cv2.resize(lbm,
                          (int(width), int(height)),
                          interpolation=cv2.INTER_NEAREST)

    def apply_to_label_map(self, label_map):
        if not self.do_padding:
            label_map = self.lbm_resize(label_map, width=self.image_size[1], height=self.image_size[0])
            if len(label_map.shape) < 3:
                label_map = np.expand_dims(label_map, axis=2)
            return label_map
        else:
            height, width, ch = label_map.shape
            if height > width:
                if height >= self.image_size[0]:
                    new_height = self.image_size[0]
                    new_width = int(width * new_height / height)
                    label_map = self.lbm_resize(label_map, width=new_width, height=new_height)
            else:
                if width >= self.image_size[1]:
                    new_width = self.image_size[1]
                    new_height = int(height * new_width / width)
                    label_map = self.lbm_resize(label_map, width=new_width, height=new_height)
            ih, iw = label_map.shape[:2]
            ph, pw = self.image_size[0], self.image_size[1]
            x = np.zeros((ph, pw, ch))
            sy1 = int(ph / 2) - int(ih / 2)
            sx1 = int(pw / 2) - int(iw / 2)
            if len(label_map.shape) < 3:
                label_map = np.expand_dims(label_map, axis=2)
            x[sy1:sy1 + ih, sx1:sx1 + iw, :] = label_map
            return x

    def un_apply(self, image):
        if self.obox is not None:
            image = image[:, self.obox[1]:self.obox[3], self.obox[0]:self.obox[2], :]
            self.obox = None
        return image


