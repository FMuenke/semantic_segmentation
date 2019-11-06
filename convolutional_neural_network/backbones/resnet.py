# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""

from keras.layers import (
    Input,
    Add,
    Activation,
    Convolution2D,
    MaxPooling2D,
    ZeroPadding2D,
)

from convolutional_neural_network.layer.fixed_batch_normalization import FixedBatchNormalization


class ResNet:
    def __init__(self, trainable=True):
        self.trainable = trainable

    def build_bottom_layers(self, x, trainable=True):
        x = ZeroPadding2D((3, 3))(x)

        x = Convolution2D(64, (7, 7), strides=(2, 2), name="conv1", trainable=trainable)(x)
        x = FixedBatchNormalization(axis=3, name="bn_conv1")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return x

    def build_residuals(self, x, trainable):
        x = conv_block(
            x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1), trainable=trainable
        )
        x = identity_block(x, 3, [64, 64, 256], stage=2, block="b", trainable=trainable)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block="c", trainable=trainable)

        x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", trainable=trainable)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="b", trainable=trainable)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="c", trainable=trainable)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="d", trainable=trainable)

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a", trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b", trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c", trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d", trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e", trainable=trainable)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f", trainable=trainable)

        x = conv_block(
            x, 3, [512, 512, 2048], stage=5, block="a", strides=(2, 2), trainable=trainable
        )
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b", trainable=trainable)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c", trainable=trainable)
        return x

    def build_backbone(self, input_layer):
        x = self.build_bottom_layers(input_layer)
        x = self.build_residuals(x, trainable=self.trainable)
        return x


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Convolution2D(
        nb_filter1, (1, 1), name=conv_name_base + "2a", trainable=trainable
    )(input_tensor)
    x = FixedBatchNormalization(axis=3, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        trainable=trainable,
    )(x)
    x = FixedBatchNormalization(axis=3, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter3, (1, 1), name=conv_name_base + "2c", trainable=trainable
    )(x)
    x = FixedBatchNormalization(axis=3, name=bn_name_base + "2c")(x)

    x = Add()([x, input_tensor])
    x = Activation("relu")(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Convolution2D(
        nb_filter1,
        (1, 1),
        strides=strides,
        name=conv_name_base + "2a",
        trainable=trainable,
    )(input_tensor)
    x = FixedBatchNormalization(axis=3, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        trainable=trainable,
    )(x)
    x = FixedBatchNormalization(axis=3, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter3, (1, 1), name=conv_name_base + "2c", trainable=trainable
    )(x)
    x = FixedBatchNormalization(axis=3, name=bn_name_base + "2c")(x)

    shortcut = Convolution2D(
        nb_filter3,
        (1, 1),
        strides=strides,
        name=conv_name_base + "1",
        trainable=trainable,
    )(input_tensor)
    shortcut = FixedBatchNormalization(axis=3, name=bn_name_base + "1")(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x