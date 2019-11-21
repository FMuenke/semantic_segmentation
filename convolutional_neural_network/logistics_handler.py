import numpy as np


class LogisticsHandler:
    def __init__(self, log_type, num_classes=None):
        self.log_type = log_type
        self.num_classes = num_classes

    def loss(self):
        if self.log_type == "fcn":
            return "binary_crossentropy"

    def decode(self, y_pred, color_coding):
        if self.log_type == "fcn":
            return self._decode_fcn(y_pred, color_coding)

        raise ValueError("{} is not known".format(self.log_type))

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
