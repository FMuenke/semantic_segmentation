import cv2
import numpy as np

from semantic_segmentation.data_structure.image_handler import ImageHandler

class Ellipse:
    def __init__(self):
        pass

    def fit(self, map):
        edg = cv2.Canny(map, threshold1=10, threshold2=30)
        h, w = edg.shape
        idx = np.where(edg>0)

        X = np.expand_dims(idx[1], axis=1)
        Y = np.expand_dims(idx[0], axis=1)
        print(X.shape)
        print(Y.shape)
        A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
        print(A.shape)
        b = np.ones_like(X)
        x = np.linalg.lstsq(A, b)[0].squeeze()
        print(
            'The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1], x[2], x[3],
                                                                                                x[4]))

        x_coord = np.linspace(0, w, w)
        y_coord = np.linspace(0, h, h)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
        print(Z_coord.shape)
        print(np.max(Z_coord))
        print(np.min(Z_coord))
        img_h = ImageHandler(np.abs(Z_coord-1))
        return img_h.normalize()

