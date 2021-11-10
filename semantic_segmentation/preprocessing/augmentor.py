import cv2
import numpy as np
from skimage.transform import rotate
'''
In this file are all functions for image augmentation stored
Important use following format for bboxes variabel!
bboxes = [bbox, bbox2, bbox3, ... ]
with bbox = [classname, x1_abs, y1_abs, x2_abs, y2_abs, (prob)]
'''


def apply_auto_contrast(img):
    for ch in range(img.shape[2]):
        equ = cv2.equalizeHist(img[:, :, ch])
        img[:, :, ch] = equ
    return img


def apply_noise(img, low=1, high=10):
    noise = np.random.randint(low=low, high=high, size=img.shape, dtype='uint8')
    img += noise
    return img


def apply_blurr(img):
    img = cv2.blur(img, (5, 5))
    return img


def apply_brightening(img):
    scale = np.random.random() + 0.5
    img = np.round(scale * img)
    return img


def apply_horizontal_flip(img, lab):
    """
    Applys a horizontal flip to image and its annotated bounding-boxes
    :param img: original image
    :param bboxes: annotated bounding-boxes
    :return:
        img: flipped image
        bboxes: flipped bboxes
    """
    img = cv2.flip(img, 1)
    lab = cv2.flip(lab, 1)
    if len(lab.shape) == 2:
        lab = np.expand_dims(lab, axis=2)
    return img, lab


def apply_vertical_flip(img, lab):
    img = cv2.flip(img, 0)
    lab = cv2.flip(lab, 0)
    if len(lab.shape) == 2:
        lab = np.expand_dims(lab, axis=2)
    return img, lab


def apply_crop(img, lab):
    height, width, ch = img.shape
    prz_zoom = 0.10
    w_random = int(width * prz_zoom)
    h_random = int(height * prz_zoom)
    if w_random > 0:
        x1_img = np.random.randint(w_random)
        x2_img = width - np.random.randint(w_random)
    else:
        x1_img = 0
        x2_img = width

    if h_random > 0:
        y1_img = np.random.randint(h_random)
        y2_img = height - np.random.randint(h_random)
    else:
        y1_img = 0
        y2_img = height

    img = img[y1_img:y2_img, x1_img:x2_img, :]
    lab = lab[y1_img:y2_img, x1_img:x2_img, :]
    return img, lab


def apply_rotation_90(img, lab):
    angle = np.random.choice([0, 90, 180, 270])

    if angle == 270:
        img = np.transpose(img, (1, 0, 2))
        img = cv2.flip(img, 0)
        lab = np.transpose(lab, (1, 0, 2))
        lab = cv2.flip(lab, 0)
        if len(lab.shape) == 2:
            lab = np.expand_dims(lab, axis=2)
    elif angle == 180:
        img = cv2.flip(img, -1)
        lab = cv2.flip(lab, -1)
        if len(lab.shape) == 2:
            lab = np.expand_dims(lab, axis=2)
    elif angle == 90:
        img = np.transpose(img, (1, 0, 2))
        img = cv2.flip(img, 1)
        lab = np.transpose(lab, (1, 0, 2))
        lab = cv2.flip(lab, 1)
        if len(lab.shape) == 2:
            lab = np.expand_dims(lab, axis=2)
    elif angle == 0:
        pass

    return img, lab


def apply_tiny_rotation(img, lab):
    img = img.astype(np.float)
    lab = lab.astype(np.float)
    rand_angle = np.random.randint(20) - 10
    img = rotate(img, angle=rand_angle, mode="reflect")
    lab = rotate(lab, angle=rand_angle, mode="reflect")
    return img.astype(np.uint8), lab.astype(np.int)


class Augmentor:
    def __init__(self):
        self.opt = {
            "horizontal_flip": True,
            "vertical_flip": True,
            "noise": True,
            "auto_contrast": False,
            "brightening": True,
            "blur": True,
            "crop": True,
            "rotation": True,
            "tiny_rotation": True,
        }

    def apply(self, img, lab):

        # Augmentation (randomized)

        if 0 == np.random.randint(6) and self.opt["horizontal_flip"]:
            img, lab = apply_horizontal_flip(img, lab)

        if 0 == np.random.randint(4) and self.opt["vertical_flip"]:
            img, lab = apply_vertical_flip(img, lab)

        if 0 == np.random.randint(5) and self.opt["noise"]:
            img = apply_noise(img)

        if 0 == np.random.randint(10) and self.opt["auto_contrast"]:
            img = apply_auto_contrast(img)

        if 0 == np.random.randint(5) and self.opt["brightening"]:
            img = apply_brightening(img)

        if 0 == np.random.randint(5) and self.opt["blur"]:
            img = apply_blurr(img)

        if 0 == np.random.randint(6) and self.opt["crop"]:
            img, lab = apply_crop(img, lab)

        if 0 == np.random.randint(6) and self.opt["tiny_rotation"]:
            img, lab = apply_tiny_rotation(img, lab)

        if 0 == np.random.randint(6) and self.opt["rotation"]:
            img, lab = apply_rotation_90(img, lab)

        if len(lab.shape) == 2:
            lab = np.expand_dims(lab, axis=2)
        return img, lab
