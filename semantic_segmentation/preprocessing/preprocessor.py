import numpy as np
from semantic_segmentation.data_structure.image_handler import ImageHandler


class Preprocessor:
    def __init__(self, image_size, padding=False):
        self.image_size = image_size
        self.min_height = 16
        self.min_width = 16
        self.do_padding = padding

    def resize(self, image):
        img_h = ImageHandler(image)
        if None in self.image_size:
            height, width = image.shape[:2]
            if height < self.min_height:
                height = self.min_height

            if width < self.min_width:
                width = self.min_width
            return img_h.resize(height=height, width=width)
        return img_h.resize(height=self.image_size[0], width=self.image_size[1])

    def normalize(self, image):
        img_h = ImageHandler(image)
        return img_h.normalize()

    def pad(self, image):
        img_h = ImageHandler(image)
        height, width = image.shape[:2]
        if None in self.image_size:
            height, width = image.shape[:2]
            if height < self.min_height:
                height = self.min_height

            if width < self.min_width:
                width = self.min_width
            return img_h.resize(height=height, width=width)
        if height > width:
            if height >= self.image_size[0]:
                new_height = self.image_size[0]
                new_width = width * self.image_size[0] / height
                image = img_h.resize(height=new_height, width=new_width)
        else:
            if width >= self.image_size[1]:
                new_width = self.image_size[1]
                new_height = height * self.image_size[0] / height
                image = img_h.resize(height=new_height, width=new_width)
        ih, iw = image.shape[:2]
        ph, pw = self.image_size[0], self.image_size[1]
        x = np.mean(image) * np.ones((ph, pw, 3))
        sy1 = int(ph/2)-int(ih/2)
        sx1 = int(pw/2)-int(iw/2)
        x[sy1:sy1+ih, sx1:sx1+iw, :] = image
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

