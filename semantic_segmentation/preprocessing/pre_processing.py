import cv2
import numpy as np
import segmentation_models as sm


def resize_img(img, input_shape):
    return cv2.resize(img, (int(input_shape[1]), int(input_shape[0])), interpolation=cv2.INTER_CUBIC)


def resize_lab(lab, input_shape):
    return cv2.resize(lab, (int(input_shape[1]), int(input_shape[0])), interpolation=cv2.INTER_NEAREST)


def pre_processing(img, input_shape, backbone):
    possible_keys = sm.get_available_backbone_names()
    img = resize_img(img, input_shape)
    selected_pre_processing = None
    for k in possible_keys:
        if k in backbone:
            selected_pre_processing = k

    if selected_pre_processing is not None:
        preproc = sm.get_preprocessing(selected_pre_processing)
        img = preproc(img)
    else:
        img = img / 128 - 0.5
    return img
