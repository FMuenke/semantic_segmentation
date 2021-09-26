import numpy as np


class LogisticsHandler:
    def __init__(self, num_classes=None, label_prep=None):
        self.num_classes = num_classes
        self.label_prep = label_prep

    def decode(self, y_pred, color_coding):
        return self._decode_fcn(y_pred, color_coding)

    def _decode_fcn(self, y_pred, color_coding):
        y_pred = y_pred[0, :, :, :]
        h, w = y_pred.shape[:2]
        color_map = np.zeros((h, w, 3))
        for i, cls in enumerate(color_coding):
            idxs = np.where(y_pred[:, :, i] > 0.5)
            color_map[idxs[0], idxs[1], :] = color_coding[cls][1]
        return color_map
