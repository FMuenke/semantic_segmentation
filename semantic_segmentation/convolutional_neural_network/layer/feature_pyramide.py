from keras.layers import Convolution2D, Add, Activation, BatchNormalization
from semantic_segmentation.convolutional_neural_network.layer.up_sample import UpSample


def feature_pyramid(lower_block, upper_block, stage):
    batch_size_lo, fmap_height_lo, fmap_width_lo, filters_lo = lower_block.get_shape().as_list()
    batch_size_up, fmap_height_up, fmap_width_up, filters_up = upper_block.get_shape().as_list()
    if not int(filters_lo) == int(filters_up):
        upper_block = Convolution2D(int(filters_lo), (1, 1), name="fp_conv_{}".format(stage))(upper_block)
        upper_block = BatchNormalization()(upper_block)
        upper_block = Activation("relu")(upper_block)
    upper_block = UpSample(name="fp_up_{}".format(stage))([upper_block, lower_block])
    fp = Add(name="fp_out_{}".format(stage))([upper_block, lower_block])
    return fp
