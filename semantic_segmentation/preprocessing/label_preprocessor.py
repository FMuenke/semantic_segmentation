import cv2
import numpy as np
from semantic_segmentation.geometric_shapes.ellipse import Ellipse


class LabelPreProcessor:
    def __init__(self, label_prep):
        self.label_prep = label_prep

    def _apply_gauss(self, label_map):
        for c in range(label_map.shape[2]):
            channel = label_map[:, :, c]
            blurred = cv2.GaussianBlur(channel, (61, 61), sigmaX=41)
            channel[channel < 1] = blurred[channel < 1]
            label_map[:, :, c] = channel
        return label_map

    def apply(self, label_map):
        h, w = label_map.shape[:2]
        if self.label_prep == "fuzzy":
            return self._apply_gauss(label_map)
            # label_map = np.reshape(label_map, (h, w, 1))
        return label_map
