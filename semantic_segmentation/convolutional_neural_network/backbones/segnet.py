from keras.layers import (
    Activation,
    BatchNormalization,
    Convolution2D,
    MaxPooling2D,
    ReLU,
    LeakyReLU,
)

from semantic_segmentation.convolutional_neural_network.layer.up_sample import UpSample


class SegNet:
    def __init__(self, num_classes, output_function="sigmoid", activation="relu"):
        self.num_classes = num_classes
        self.out_f = output_function
        self.activation = activation

    def act_unit(self, x):
        if self.activation == "relu":
            return ReLU()(x)
        if self.activation == "leaky_relu":
            return LeakyReLU()(x)
        raise ValueError("Unknown activation option: {}".format(self.activation))

    def enc_unit(self, x, num_filter, stage):
        conv = Convolution2D(num_filter, kernel_size=3, kernel_initializer='he_normal', padding="same",
                             name="segnet_conv_{}".format(stage))(x)

        conv = BatchNormalization()(conv)
        conv = self.act_unit(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return pool, conv

    def dec_unit(self, confh, confl, num_filter, stage):
        conv = UpSample()([confh, confl])
        conv = Convolution2D(num_filter, kernel_size=3, kernel_initializer='he_normal', padding="same",
                             name="segnet_conv_{}".format(stage))(conv)

        conv = BatchNormalization()(conv)
        conv = self.act_unit(conv)
        return conv

    def lat_unit(self, x, num_filter, stage):
        conv = Convolution2D(num_filter, kernel_size=3, kernel_initializer='he_normal', padding="same",
                             name="segnet_conv_{}".format(stage))(x)

        conv = BatchNormalization()(conv)
        conv = self.act_unit(conv)
        return conv

    def build(self, input_tensor):
        pool1, conv1 = self.enc_unit(input_tensor, 64, "1_b_model")
        pool2, conv2 = self.enc_unit(pool1, 128, "2_b_model")
        pool3, conv3 = self.enc_unit(pool2, 256, "3_b_model")

        conv4 = self.lat_unit(pool3, 512, "4_b_model")
        conv5 = self.lat_unit(conv4, 512, "5_b_model")

        conv6 = self.dec_unit(conv5, conv3, 256, "6_b_model")
        conv7 = self.dec_unit(conv6, conv2, 128, "7_b_model")
        conv8 = self.dec_unit(conv7, conv1, 64, "8_b_model")

        x = Convolution2D(self.num_classes, kernel_size=3, padding="same", name="segnet_classification")(conv8)
        out = Activation(self.out_f)(x)
        return out
