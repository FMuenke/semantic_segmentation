import cv2
import numpy as np
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops
from semantic_segmentation.data_structure.image_handler import ImageHandler


class Ellipse:
    def __init__(self, map):
        self.array_size = map.shape[:2]
        self.x = self.fit_from_map(map)

        self.prop = self._init_properties()

    def __str__(self):
        return "Ellipse defined by {0:.3}x^2+{1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1".format(self.x[0],
                                                                                            self.x[1],
                                                                                            self.x[2],
                                                                                            self.x[3],
                                                                                            self.x[4])

    def _init_properties(self):
        prop = {}
        map = self.map_to_array()
        regions = regionprops(map.as_type(np.int))
        prop["centroid"] = regions[0].centroid
        prop["orientation"] = regions[0].orientation
        prop["major_axis_length"] = regions[0].major_axis_length
        prop["minor_axis_length"] = regions[0].minor_axis_length
        return regions[0]

    def fit_from_map(self, map):
        if len(map.shape) == 3:
            map = np.sum(map, axis=2)
        map[map > 0] = 1
        kernel = (2, 2)
        edg = cv2.morphologyEx(map, cv2.MORPH_GRADIENT, kernel)
        idx = np.where(edg > 0)
        X = np.expand_dims(idx[1], axis=1)
        Y = np.expand_dims(idx[0], axis=1)
        A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
        b = np.ones_like(X)
        return np.linalg.lstsq(A, b)[0].squeeze()

    def map_to_array(self, array_size=None):
        if array_size is None:
            array_size = self.array_size
        h, w = array_size
        x_coord = np.linspace(0, w, w)
        y_coord = np.linspace(0, h, h)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = self.x[0] * X_coord ** 2 + self.x[1] * X_coord * Y_coord + self.x[2] * Y_coord ** 2 + self.x[3] * X_coord + self.x[4] * Y_coord
        Z_coord[Z_coord < 1] = 0
        Z_coord[Z_coord >= 1] = 1
        return Z_coord
        # img_h = ImageHandler(Z_coord)
        # return img_h.normalize() / 255

    def build_label_map(self, array_size=None):
        if array_size is None:
            array_size = self.array_size
        cv2.GaussianBlur(label_map[:, :, c], (61, 61), sigmaX=41)


    def get_center(self):
        pass

    def compare_to_ellipse(self, ellipse):
        center_diff = np.sqrt((self.prop["centroid"][0] - ellipse.prop["centroid"][0]) ** 2
                              + (self.prop["centroid"][1] - ellipse.prop["centroid"][1]) ** 2)

        orientation_diff = np.square(self.prop["orientation"] - self.prop["orientation"])
        minor_axis_length_diff = np.square(self.prop["minor_axis_length"] - ellipse.prop["minor_axis_length"])
        major_axis_length_diff = np.square(self.prop["major_axis_length"] - ellipse.prop["major_axis_length"])

        return [center_diff, orientation_diff, major_axis_length_diff, minor_axis_length_diff]
