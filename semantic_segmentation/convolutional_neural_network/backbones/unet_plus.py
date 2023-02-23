from tensorflow.keras.layers import (
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
    SeparableConv2D,
    Concatenate,
    LayerNormalization,
    ReLU,
    Activation,
    Conv2DTranspose,
)
import tensorflow as tf


def norm_activation(x):
    x = LayerNormalization()(x)
    x = Activation(activation=tf.nn.gelu)(x)
    return x


class UNetPlus:
    def __init__(self, num_classes, output_function="sigmoid"):
        self.num_classes = num_classes
        self.out_f = output_function

    def enc_unit(self, x, num_filter, stage):
        conv = SeparableConv2D(
            num_filter, 3, padding='same', kernel_initializer='he_normal',
            name="unet_conv{}1".format(stage))(x)
        conv = norm_activation(conv)
        conv = SeparableConv2D(
            num_filter, 3, padding='same', kernel_initializer='he_normal',
            name="unet_conv{}2".format(stage))(conv)
        conv = norm_activation(conv)
        pool = MaxPooling2D()(conv)
        return pool, conv

    def dec_unit(self, convh, convl, num_filter, stage):
        up = UpSampling2D()(convh)
        up = SeparableConv2D(
            num_filter, 2, padding='same', kernel_initializer='he_normal',
            name="unet_up{}0".format(stage))(up)
        up = norm_activation(up)
        merge = Concatenate(axis=3)([convl, up])
        conv = SeparableConv2D(
            num_filter, 3, padding='same', kernel_initializer='he_normal',
            name="unet_up{}1".format(stage))(merge)
        conv = norm_activation(conv)
        return conv

    def build(self, input_tensor):
        # pool0 = Convolution2D(8, (7, 7), padding="same")(input_tensor)
        pool1, conv1 = self.enc_unit(input_tensor, 64, stage=1)
        pool2, conv2 = self.enc_unit(pool1, 128, stage=2)
        pool3, conv3 = self.enc_unit(pool2, 256, stage=3)
        pool4, conv4 = self.enc_unit(pool3, 512, stage=4)

        pool5, conv5 = self.enc_unit(pool4, 1024, stage=5)
        conv6 = self.dec_unit(conv5, conv4, 512, stage=6)

        conv7 = self.dec_unit(conv6, conv3, 256, stage=7)
        conv8 = self.dec_unit(conv7, conv2, 128, stage=8)
        conv9 = self.dec_unit(conv8, conv1, 64, stage=9)

        if self.out_f is not None:
            out = SeparableConv2D(
                self.num_classes,
                1,
                activation=self.out_f,
                padding='same',
                kernel_initializer='uniform',
                name="unet_classification_layer")(conv9)
        else:
            out = conv9
        return out
