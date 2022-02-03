import os
from PIL import Image
import cv2


def get_file_name(base_path, data_id, extensions):
    for ext in extensions:
        filename = os.path.join(base_path, data_id + ext)
        if os.path.isfile(filename):
            return filename
    return None


class SegmentationData:

    image_extensions = [".jpg", ".JPG", ".png", "PNG", ".jpeg"]
    label_extensions = [".png", ".PNG", "_label.tif", "_label.tiff"]

    def __init__(self, base_path, data_id, label_src="labels"):
        self.id = data_id
        self.path = base_path

        self.image_path = os.path.join(base_path, "images")
        self.label_path = os.path.join(base_path, label_src)

        self.image_file = get_file_name(self.image_path, self.id, self.image_extensions)
        self.label_file = get_file_name(self.label_path, self.id, self.label_extensions)

    def get_image_size(self):
        if self.image_file is None:
            raise Exception("NO IMAGE FILE AVAILABLE")
        im = Image.open(self.image_file)
        width, height = im.size
        return height, width

    def load_y(self):
        return cv2.imread(self.label_file)