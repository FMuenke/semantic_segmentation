from keras.layers import (
    Activation,
    BatchNormalization,
    Convolution2D,
    MaxPooling2D,
    ReLU,
    LeakyReLU,
)

from semantic_segmentation.convolutional_neural_network.layer.up_sample import UpSample
from semantic_segmentation.convolutional_neural_network.layer.pool_and_unpool import MaxPoolingWithArgmax2D, MaxUnpooling2D


class SegNet:
    def __init__(self, num_classes, output_function="sigmoid", activation="relu"):
        self.num_classes = num_classes
        self.out_f = output_function
        self.activation = activation
        self.pool_size = (2, 2)

    def act_unit(self, x):
        if self.activation == "relu":
            return ReLU()(x)
        if self.activation == "leaky_relu":
            return LeakyReLU()(x)
        raise ValueError("Unknown activation option: {}".format(self.activation))

    def conv_block(self, x, num_filter, stage):
        for i, fn in enumerate(num_filter):
            x = Convolution2D(fn, kernel_size=3, kernel_initializer='he_normal', padding="same",
                              name="segnet_conv_{}_{}".format(stage, i))(x)
            x = BatchNormalization()(x)
            x = self.act_unit(x)
        return x

    def enc_unit(self, x, num_filter, stage):
        conv = self.conv_block(x, num_filter, stage)
        pool, mask = MaxPoolingWithArgmax2D(self.pool_size)(conv)
        return pool, mask

    def dec_unit(self, conv, mask, num_filter, stage):
        conv = MaxUnpooling2D(self.pool_size)([conv, mask])
        conv = self.conv_block(conv, num_filter, stage)
        return conv

    def lat_unit(self, x, num_filter, stage):
        conv = self.conv_block(x, num_filter, stage)
        return conv

    def build(self, input_tensor):
        pool1, mask1 = self.enc_unit(input_tensor, [64, 64], "enc_1")
        pool2, mask2 = self.enc_unit(pool1, [128, 128], "enc_2")
        pool3, mask3 = self.enc_unit(pool2, [256, 256, 256], "enc_3")
        pool4, mask4 = self.enc_unit(pool3, [512, 512, 512], "enc_4")
        pool5, mask5 = self.enc_unit(pool4, [512, 512, 512], "enc_5")

        un_pool1 = self.dec_unit(pool5, mask5, [512, 512, 512], "dec_1")
        un_pool2 = self.dec_unit(un_pool1, mask4, [512, 512, 256], "dec_2")
        un_pool3 = self.dec_unit(un_pool2, mask3, [256, 256, 128], "dec_3")
        un_pool4 = self.dec_unit(un_pool3, mask2, [128, 64], "dec_4")
        un_pool5 = self.dec_unit(un_pool4, mask1, [64], "dec_5")

        out = Convolution2D(self.num_classes, (1, 1), padding="valid", name="segnet_classification_conv")(un_pool5)
        out = Activation(self.out_f, name="segnet_classification")(out)
        return out
