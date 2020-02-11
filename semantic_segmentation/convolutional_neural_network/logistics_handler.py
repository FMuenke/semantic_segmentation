import numpy as np
from semantic_segmentation.convolutional_neural_network.losses import dice, focal_loss, jaccard, weighted_cross_entropy, mixed


class LogisticsHandler:
    def __init__(self, loss_type, num_classes=None):
        self.loss_type = loss_type
        self.num_classes = num_classes

    def loss(self):
        if self.loss_type == "focal_loss":
            return focal_loss()
        if self.loss_type == "binary_crossentropy":
            return "binary_crossentropy"
        if self.loss_type == "dice":
            return dice()
        if self.loss_type == "jaccard":
            return jaccard()
        if self.loss_type == "weighted_cross_entropy":
            return weighted_cross_entropy(100)
        if self.loss_type == "mixed":
            return mixed()

        raise ValueError("Loss Type unknown {}".format(self.loss_type))

    def decode(self, y_pred, color_coding):
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
