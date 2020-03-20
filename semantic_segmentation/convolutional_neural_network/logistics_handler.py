import numpy as np
from semantic_segmentation.data_structure.image_handler import ImageHandler


class LogisticsHandler:
    def __init__(self, num_classes=None, label_prep=None):
        self.num_classes = num_classes
        self.label_prep = label_prep

    def decode(self, y_pred, color_coding):
        if self.label_prep is None or self.label_prep == "basic":
            return self._decode_fcn(y_pred, color_coding)
        if self.label_prep == "ellipse":
            return self._decode_ellipse(y_pred)
        raise ValueError("Label Prep: {}, Not Known!".format(self.label_prep))

    def _decode_fcn(self, y_pred, color_coding):
        y_pred = y_pred[0, :, :, :]
        h, w = y_pred.shape[:2]
        color_map = np.zeros((h, w, 3))
        for x in range(w):
            for y in range(h):
                for i, cls in enumerate(color_coding):
                    if y_pred[y, x, i] > 0.5:
                        color_map[y, x, :] = color_coding[cls][1]
        return color_map

    def _decode_ellipse(self, y_pred):
        b, h, w, c = y_pred.shape
        y_pred = y_pred[0, :, :, :]
        # y_pred = np.reshape(y_pred, (h, w, 1))
        # y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
        img_h = ImageHandler(y_pred)
        i_norm = img_h.normalize()
        return i_norm
