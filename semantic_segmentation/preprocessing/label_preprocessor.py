import cv2
import numpy as np
from semantic_segmentation.geometric_shapes.ellipse import Ellipse


class LabelPreProcessor:
    def __init__(self, label_prep):
        self.label_prep = label_prep

    def _apply_gauss(self, label_map):
        for c in range(label_map.shape[2]):
            label_map[:, :, c] = cv2.GaussianBlur(label_map[:, :, c], (61, 61), sigmaX=41)
        return label_map

    def _apply_for_ellipse(self, label_map):
        e = Ellipse(label_map)
        param = [e.prop["centroid"][0],
                 e.prop["centroid"][1],
                 e.prop["orientation"],
                 e.prop["major_axis_length"],
                 e.prop["minor_axis_length"]]
        return [label_map, np.array(param)]

    def apply(self, label_map):
        h, w = label_map.shape[:2]
        if self.label_prep == "fuzzy":
            return self._apply_gauss(label_map)
        if self.label_prep == "ellipse":
            return self._apply_for_ellipse(label_map)
            # label_map = np.reshape(label_map, (h, w, 1))
        return label_map, None
