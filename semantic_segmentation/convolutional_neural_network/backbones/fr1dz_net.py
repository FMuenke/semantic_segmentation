from keras.layers import (
    Activation,
    Convolution2D,
    MaxPooling2D,
    Add,
)

from semantic_segmentation.convolutional_neural_network.layer.feature_pyramide import feature_pyramid
from semantic_segmentation.convolutional_neural_network.layer.fixed_batch_normalization import FixedBatchNormalization


class Fr1dzNet:
    def __init__(self, num_classes, output_function="sigmoid"):
        self.num_classes = num_classes
        self.out_f = output_function

    def build(self, input_tensor):
        s0 = input_tensor
        # Encoder
        s1 = conv_block(s0, 3, [64, 64, 256], stage=1, block="a", strides=(1, 1))
        s1 = identity_block(s1, 3, [64, 64, 256], stage=1, block="b")
        s1 = identity_block(s1, 3, [64, 64, 256], stage=1, block="c")

        s2 = conv_block(s1, 3, [128, 128, 512], stage=2, block="a")
        s2 = identity_block(s2, 3, [128, 128, 512], stage=2, block="b")
        s2 = identity_block(s2, 3, [128, 128, 512], stage=2, block="c")
        s2 = identity_block(s2, 3, [128, 128, 512], stage=2, block="d")

        s3 = conv_block(s2, 3, [256, 256, 1024], stage=3, block="a")
        s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="b")
        s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="c")
        s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="d")
        s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="e")
        s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="f")

        s4 = conv_block(s3, 3, [512, 512, 2048], stage=4, block="a", strides=(2, 2))
        s4 = identity_block(s4, 3, [512, 512, 2048], stage=4, block="b")
        s4 = identity_block(s4, 3, [512, 512, 2048], stage=4, block="c")

        # Latent Space

        l = Convolution2D(2048, (3, 3), name="f1_latent")(s4)
        l = MaxPooling2D((2, 2))(l)

        # Decoder
        f4 = feature_pyramid(lower_block=s4, upper_block=l, stage=4)

        f3 = feature_pyramid(lower_block=s3, upper_block=f4, stage=3)

        f2 = feature_pyramid(lower_block=s2, upper_block=f3, stage=2)

        f1 = feature_pyramid(lower_block=s1, upper_block=f2, stage=1)

        x = f1

        x = Convolution2D(self.num_classes, kernel_size=3, padding="same", name="fr1dz_classification")(x)
        out = Activation(self.out_f)(x)
        return out


def conv_block(
    input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True
):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = "f1_res" + str(stage) + block + "_branch"
    bn_name_base = "f1_bn" + str(stage) + block + "_branch"

    x = Convolution2D(
        nb_filter1,
        (1, 1),
        strides=strides,
        name=conv_name_base + "2a",
        trainable=trainable,
    )(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        trainable=trainable,
    )(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter3, (1, 1), name=conv_name_base + "2c", trainable=trainable
    )(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    shortcut = Convolution2D(
        nb_filter3,
        (1, 1),
        strides=strides,
        name=conv_name_base + "1",
        trainable=trainable,
    )(input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = "f1_res" + str(stage) + block + "_branch"
    bn_name_base = "f1_bn" + str(stage) + block + "_branch"

    x = Convolution2D(
        nb_filter1, (1, 1), name=conv_name_base + "2a", trainable=trainable
    )(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        trainable=trainable,
    )(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter3, (1, 1), name=conv_name_base + "2c", trainable=trainable
    )(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + "2c")(x)

    x = Add()([x, input_tensor])
    x = Activation("relu")(x)
    return x
