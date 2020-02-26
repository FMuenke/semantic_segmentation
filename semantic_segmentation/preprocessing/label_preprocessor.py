import cv2
from semantic_segmentation.preprocessing.ellipse import Ellipse


class LabelPreProcessor:
    def __init__(self):
        pass

    def _apply_gauss(self, label_map):
        for c in range(label_map.shape[2]):
            label_map[:, :, c] = cv2.GaussianBlur(label_map[:, :, c], (61, 61), sigmaX=41)
        return label_map

    def _apply_for_ellipse(self, label_map):
        e = Ellipse()
        label_map = e.fit(label_map)
        return label_map

    def apply(self, label_map):
        # label_map = self._apply_gauss(label_map)
        label_map = self._apply_for_ellipse(label_map)
        return label_map
