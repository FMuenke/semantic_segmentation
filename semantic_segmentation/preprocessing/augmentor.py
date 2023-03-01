import cv2
import numpy as np
from skimage.transform import rotate
'''
In this file are all functions for image augmentation stored
Important use following format for bboxes variabel!
bboxes = [bbox, bbox2, bbox3, ... ]
with bbox = [classname, x1_abs, y1_abs, x2_abs, y2_abs, (prob)]
'''

class ChannelShift:
    def __init__(self, intensity, seed=2022):
        self.name = "ChannelShift"
        assert 1 < intensity < 255, "Set the pixel values to be shifted (1, 255)"
        self.intensity = intensity
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        height, width, ch = img.shape
        img = img.astype(np.float32)
        for i in range(ch):
            img[:, :, i] += self.rng.integers(self.intensity) * self.rng.choice([1, -1])
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)


class Stripes:
    def __init__(self, horizontal, vertical, space, width, intensity):
        self.name = "Stripes"
        self.horizontal = horizontal
        self.vertical = vertical
        self.space = space
        self.width = width
        self.intensity = intensity

    def apply(self, img):
        h, w, c = img.shape
        g_h = int(h / self.width)
        g_w = int(w / self.width)
        mask = np.zeros([g_h, g_w, c])

        if self.horizontal:
            mask[::self.space, :, :] = self.intensity
        if self.vertical:
            mask[:, ::self.space, :] = self.intensity

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img = mask.astype(np.float32) + img.astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


class Blurring:
    def __init__(self, kernel=9, randomness=-1, seed=2022):
        self.name = "Blurring"
        if randomness == -1:
            randomness = kernel - 2
        assert 0 < randomness < kernel, "REQUIREMENT: 0 < randomness ({}) < kernel({})".format(randomness, kernel)
        self.kernel = kernel
        self.randomness = randomness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        k = self.kernel + self.rng.integers(-self.randomness, self.randomness)
        img = cv2.blur(img.astype(np.float32), ksize=(k, k))
        return img.astype(np.uint8)


class NeedsMoreJPG:
    def __init__(self, percentage, randomness, seed=2022):
        self.name = "NeedsMoreJPG"
        self.percentage = percentage
        self.randomness = randomness
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(self, img):
        h, w = img.shape[:2]
        p = self.percentage + self.rng.integers(-self.randomness, self.randomness)
        img = cv2.resize(img, (int(w * p / 100), int(h * p / 100)), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img


class SaltNPepper:
    def __init__(self, max_delta, grain_size):
        self.name = "SaltNPepper"
        self.max_delta = max_delta
        self.grain_size = grain_size

    def apply(self, img):
        h, w, c = img.shape
        snp_h = max(int(h / self.grain_size), 3)
        snp_w = max(int(w / self.grain_size), 3)
        snp = np.random.randint(-self.max_delta, self.max_delta, size=[snp_h, snp_w, c])
        snp = cv2.resize(snp, (w, h), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.int32) + snp
        return np.clip(img, 0, 255).astype(np.uint8)


def apply_auto_contrast(img):
    for ch in range(img.shape[2]):
        equ = cv2.equalizeHist(img[:, :, ch])
        img[:, :, ch] = equ
    return img


def apply_noise(img):
    noise = SaltNPepper(
        max_delta=np.random.choice([5, 10, 15]),
        grain_size=np.random.choice([1, 2, 4, 8])
    )
    return noise.apply(img)


def apply_blur(img):
    noise = Blurring(kernel=9, randomness=5)
    return noise.apply(img)


def apply_channel_shift(img):
    noise = ChannelShift(intensity=np.random.choice([5, 10, 15]))
    return noise.apply(img)


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
            "horizontal_flip": False,
            "vertical_flip": False,
            "crop": True,
            "rotation": True,
            "tiny_rotation": False,

            "noise": True,
            "brightening": True,
            "blur": True,
        }

    def apply(self, img, lab):

        # Augmentation (randomized)

        if 0 == np.random.randint(6) and self.opt["horizontal_flip"]:
            img, lab = apply_horizontal_flip(img, lab)

        if 0 == np.random.randint(4) and self.opt["vertical_flip"]:
            img, lab = apply_vertical_flip(img, lab)

        if 0 == np.random.randint(5) and self.opt["noise"]:
            img = apply_noise(img)

        if 0 == np.random.randint(5) and self.opt["brightening"]:
            img = apply_channel_shift(img)

        if 0 == np.random.randint(5) and self.opt["blur"]:
            img = apply_blur(img)

        if 0 == np.random.randint(6) and self.opt["crop"]:
            img, lab = apply_crop(img, lab)

        if 0 == np.random.randint(6) and self.opt["tiny_rotation"]:
            img, lab = apply_tiny_rotation(img, lab)

        if 0 == np.random.randint(6) and self.opt["rotation"]:
            img, lab = apply_rotation_90(img, lab)

        if len(lab.shape) == 2:
            lab = np.expand_dims(lab, axis=2)
        return img, lab
