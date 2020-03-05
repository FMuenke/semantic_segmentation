import cv2
import numpy as np

from semantic_segmentation.data_structure.image_handler import ImageHandler


class Ellipse:
    def __init__(self, x=None):
        self.x = x

    def __str__(self):
        return "Ellipse defined by {0:.3}x^2+{1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1".format(self.x[0],
                                                                                            self.x[1],
                                                                                            self.x[2],
                                                                                            self.x[3],
                                                                                            self.x[4])

    def fit_from_map(self, map):
        map *= 255
        edg = cv2.Canny(map, threshold1=10, threshold2=30)
        idx = np.where(edg > 0)
        X = np.expand_dims(idx[1], axis=1)
        Y = np.expand_dims(idx[0], axis=1)
        A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
        b = np.ones_like(X)
        self.x = np.linalg.lstsq(A, b)[0].squeeze()

    def map_to_array(self, array_size):
        h, w = array_size
        x_coord = np.linspace(0, w, w)
        y_coord = np.linspace(0, h, h)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = self.x[0] * X_coord ** 2 + self.x[1] * X_coord * Y_coord + self.x[2] * Y_coord ** 2 + self.x[3] * X_coord + self.x[4] * Y_coord
        Z_coord[Z_coord < 1] = 0
        img_h = ImageHandler(Z_coord)
        return img_h.normalize() / 255
