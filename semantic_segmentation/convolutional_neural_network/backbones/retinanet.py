from keras.layers import (
    Activation,
    Convolution2D,
    MaxPooling2D,
    ZeroPadding2D,
    AveragePooling2D,
)


from neural_network.Layers.FixedBatchNormalization import FixedBatchNormalization
from neural_network.Layers.feature_pyramide import feature_pyramid
from neural_network.Layers.pre_predictor import pre_predictor

from neural_network.Layers.layer_blocks import conv_block, identity_block


def backbone(input_tensor=None, trainable=False):

    bn_axis = 3

    x = ZeroPadding2D((3, 3))(input_tensor)

    x = Convolution2D(64, (7, 7), strides=(2, 2), name="conv1", trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(
        x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1), trainable=trainable
    )
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b", trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c", trainable=trainable)

    block_4 = x

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b", trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c", trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="d", trainable=trainable)

    block_3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block="a", trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="b", trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="c", trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="d", trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="e", trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block="f", trainable=trainable)

    block_2 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block="a", strides=(2, 2), trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="b", trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block="c", trainable=trainable)

    block_1 = x

    x = AveragePooling2D((7, 7), name="avg_pool")(x)

    p_5 = block_1
    p_4 = feature_pyramid(block_2, p_5, stage=4)
    p_3 = feature_pyramid(block_3, p_4, stage=3)

    box_1 = pre_predictor(p_5, stage=1, num_filters=1024)
    box_2 = pre_predictor(p_4, stage=2, num_filters=512)
    box_3 = pre_predictor(p_3, stage=3, num_filters=256)

    return x, [box_1, box_2, box_3], None
