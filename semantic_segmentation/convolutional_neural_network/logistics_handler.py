import numpy as np
from semantic_segmentation.data_structure.image_handler import ImageHandler
from semantic_segmentation.convolutional_neural_network.losses import dice, focal_loss, jaccard, weighted_cross_entropy, mixed


class LogisticsHandler:
    def __init__(self, loss_type, num_classes=None):
        self.loss_type = loss_type
        self.num_classes = num_classes

    def loss(self):
        if self.loss_type in ["focal", "focal_loss"]:
            return focal_loss()
        if self.loss_type in ["binary_crossentropy", "bc"]:
            return "binary_crossentropy"
        if self.loss_type == "dice":
            return dice()
        if self.loss_type == "jaccard":
            return jaccard()
        if self.loss_type == "weighted_cross_entropy":
            return weighted_cross_entropy(100)
        if self.loss_type == "mixed":
            return mixed()
        if self.loss_type in ["mean_squared_error", "mse"]:
            return "mean_squared_error"

        raise ValueError("Loss Type unknown {}".format(self.loss_type))

    def decode(self, y_pred, color_coding):
        if self.loss_type in ["mean_squared_error", "mse"]:
            return self._decode_ellipse(y_pred)
        return self._decode_fcn(y_pred, color_coding)

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
