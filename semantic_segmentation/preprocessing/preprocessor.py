from semantic_segmentation.data_structure.image_handler import ImageHandler


class Preprocessor:
    def __init__(self, image_size):
        self.image_size = image_size
        self.min_height = 16
        self.min_width = 16

    def resize(self, image):
        img_h = ImageHandler(image)
        if self.image_size[2] == 1:
            image = img_h.gray()
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

    def apply(self, image):
        image = self.resize(image)
        image = self.normalize(image)
        return image

