from tensorflow.keras.layers import (
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    Concatenate,
    LeakyReLU,
    ReLU,
)

from semantic_segmentation.convolutional_neural_network.layer.up_sample import UpSample


class UNet:
    def __init__(self, num_classes,
                 output_function="sigmoid",
                 dyn_up_sampling=True,
                 batch_norm=True,
                 reduced=False,
                 activation="relu"):
        self.num_classes = num_classes
        self.out_f = output_function
        self.dyn_up_sampling = dyn_up_sampling
        self.batch_norm = batch_norm
        self.reduced = reduced
        self.activation = activation

    def act_unit(self, x):
        if self.activation == "relu":
            return ReLU()(x)
        if self.activation == "leaky_relu":
            return LeakyReLU()(x)
        raise ValueError("Unknown activation option: {}".format(self.activation))

    def enc_unit(self, x, num_filter, stage):
        conv = Convolution2D(num_filter, 3, padding='same', kernel_initializer='he_normal',
                             name="unet_conv{}1".format(stage))(x)
        conv = self.act_unit(conv)
        if self.batch_norm:
            conv = BatchNormalization()(conv)
        conv = Convolution2D(num_filter, 3, padding='same', kernel_initializer='he_normal',
                             name="unet_conv{}2".format(stage))(conv)
        conv = self.act_unit(conv)
        if self.batch_norm:
            conv = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return pool, conv

    def dec_unit(self, convh, convl, num_filter, stage):
        if self.dyn_up_sampling:
            up = UpSample()([convh, convl])
        else:
            up = UpSampling2D()(convh)
        up = Convolution2D(num_filter, 2, padding='same', kernel_initializer='he_normal',
                           name="unet_up{}0".format(stage))(up)
        up = self.act_unit(up)
        merge = Concatenate(axis=3)([convl, up])
        conv = Convolution2D(num_filter, 3, padding='same', kernel_initializer='he_normal',
                             name="unet_up{}1".format(stage))(merge)
        conv = self.act_unit(conv)
        if self.batch_norm:
            conv = BatchNormalization()(conv)
        conv = Convolution2D(num_filter, 3, padding='same', kernel_initializer='he_normal',
                             name="unet_up{}2".format(stage))(conv)
        conv = self.act_unit(conv)
        if self.batch_norm:
            conv = BatchNormalization()(conv)
        return conv

    def build(self, input_tensor):

        pool1, conv1 = self.enc_unit(input_tensor, 64, stage=1)
        pool2, conv2 = self.enc_unit(pool1, 128, stage=2)
        pool3, conv3 = self.enc_unit(pool2, 256, stage=3)
        pool4, conv4 = self.enc_unit(pool3, 512, stage=4)

        if not self.reduced:
            pool5, conv5 = self.enc_unit(pool4, 1024, stage=5)
            conv6 = self.dec_unit(conv5, conv4, 512, stage=6)
        else:
            conv6 = conv4

        conv7 = self.dec_unit(conv6, conv3, 256, stage=7)
        conv8 = self.dec_unit(conv7, conv2, 128, stage=8)
        conv9 = self.dec_unit(conv8, conv1, 64, stage=9)

        if self.out_f is not None:
            out = Convolution2D(self.num_classes,
                                3,
                                activation=self.out_f,
                                padding='same',
                                kernel_initializer='uniform',
                                name="unet_classification_layer")(conv9)
        else:
            out = conv9
        return out
