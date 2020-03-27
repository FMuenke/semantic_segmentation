from tensorflow.keras.layers import (
    Activation,
    Convolution2D,
)

from semantic_segmentation.convolutional_neural_network.layer.feature_pyramide import feature_pyramid
from semantic_segmentation.convolutional_neural_network.layer.resnet_blocks import conv_block, identity_block


class Fr1dzNet:
    def __init__(self, num_classes, output_function="sigmoid"):
        self.num_classes = num_classes
        self.out_f = output_function

    def build(self, input_tensor):
        s0 = input_tensor
        s0 = Convolution2D(filters=32, kernel_size=(2, 2), padding="same", name="stage_0_a")(s0)
        # Encoder
        s1 = conv_block(s0, 3, [64, 64, 256], stage=1, block="a", strides=(2, 2))
        s1 = identity_block(s1, 3, [64, 64, 256], stage=1, block="b")
        # s1 = identity_block(s1, 3, [64, 64, 256], stage=1, block="c")

        s2 = conv_block(s1, 3, [128, 128, 512], stage=2, block="a", strides=(2, 2))
        s2 = identity_block(s2, 3, [128, 128, 512], stage=2, block="b")
        # s2 = identity_block(s2, 3, [128, 128, 512], stage=2, block="c")
        # s2 = identity_block(s2, 3, [128, 128, 512], stage=2, block="d")

        s3 = conv_block(s2, 3, [256, 256, 1024], stage=3, block="a", strides=(2, 2))
        s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="b")
        # s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="c")
        # s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="d")
        # s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="e")
        # s3 = identity_block(s3, 3, [256, 256, 1024], stage=3, block="f")

        # s4 = conv_block(s3, 3, [512, 512, 2048], stage=4, block="a", strides=(2, 2))
        # s4 = identity_block(s4, 3, [512, 512, 2048], stage=4, block="b")
        # s4 = identity_block(s4, 3, [512, 512, 2048], stage=4, block="c")

        # Decoder
        # f4 = feature_pyramid(lower_block=s4, upper_block=s3, stage=4)

        f3 = feature_pyramid(lower_block=s2, upper_block=s3, stage=3)

        f2 = feature_pyramid(lower_block=s1, upper_block=f3, stage=2)

        f1 = feature_pyramid(lower_block=s0, upper_block=f2, stage=1)

        x = f1

        x = Convolution2D(self.num_classes, kernel_size=3, padding="same", name="fr1dz_classification")(x)
        out = Activation(self.out_f)(x)
        return out


