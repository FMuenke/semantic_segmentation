import numpy as np

from keras.layers import Convolution2D, Dense, Flatten, Concatenate
from convolutional_neural_network.layer.up_sample import UpSample


class LogisticsHandler:
    def __init__(self, log_type, num_classes=None):
        self.log_type = log_type
        self.num_classes = num_classes

    def attach(self, x):
        if self.log_type == "fcn":
            assert self.num_classes is not None, "Number of classes needs to be given"
            x = Convolution2D(self.num_classes, (1, 1), activation="sigmoid")(x)
            return UpSample()(x)

        raise ValueError("{} is not known".format(self.log_type))

    def loss(self):
        if self.log_type == "fcn":
            return "binary_crossentropy"

    def decode(self, y_pred):
        if self.log_type == "fcn":
            return self._decode_fcn(y_pred)

        raise ValueError("{} is not known".format(self.log_type))

    def _decode_fcn(self, y_pred):
        class_idx = np.argmax(y_pred, axis=1)
        confidence = np.max(y_pred, axis=1)
        return class_idx
