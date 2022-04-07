from tensorflow.keras.layers import (
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
    Concatenate,
    LeakyReLU,
    ReLU,
    Activation,
    Add
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations

from semantic_segmentation.convolutional_neural_network.layer.up_sample import UpSample


class ResidualUNet:
    def __init__(self, num_classes,
                 output_function="sigmoid",
                 dyn_up_sampling=False,
                 batch_norm=True):
        self.num_classes = num_classes
        self.out_f = output_function
        self.dyn_up_sampling = dyn_up_sampling
        self.batch_norm = batch_norm
        self.reg = 0.001

    def res_identity(self, x, filters):
        x_skip = x
        f1, f2 = filters

        # first block
        x = Convolution2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.reg))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # second block # bottleneck (but size kept same with padding)
        x = Convolution2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.reg))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # third block activation used after adding the input
        x = Convolution2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.reg))(x)
        x = BatchNormalization()(x)
        # x = Activation(activations.relu)(x)

        # add the input
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)
        return x

    def res_conv(self, x, s, filters):
        x_skip = x
        f1, f2 = filters

        # first block
        x = Convolution2D(f1, (1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(self.reg))(x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        # second block
        x = Convolution2D(f1, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(self.reg))(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)

        # third block
        x = Convolution2D(f2, (1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(self.reg))(x)
        x = BatchNormalization()(x)

        # shortcut
        x_skip = Convolution2D(f2, (1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add
        x = Add()([x, x_skip])
        x = Activation(activations.relu)(x)
        return x

    def enc_unit(self, x, num_filter, stage):
        f1, f2 = num_filter
        conv_skip = x
        conv = Convolution2D(
            f1,
            1,
            padding='same',
            kernel_initializer='he_normal',
            name="unet_conv{}1".format(stage)
        )(x)
        conv = Activation(activations.relu)(conv)
        conv = BatchNormalization()(conv)

        conv = Convolution2D(
            f1,
            3,
            padding='same',
            kernel_initializer='he_normal',
            name="unet_conv{}2".format(stage)
        )(conv)
        conv = Activation(activations.relu)(conv)
        conv = BatchNormalization()(conv)

        conv = Convolution2D(
            f2,
            1,
            padding='same',
            kernel_initializer='he_normal',
            name="unet_conv{}3".format(stage)
        )(conv)
        conv = Activation(activations.relu)(conv)
        conv = BatchNormalization()(conv)

        conv = Add()([conv, conv_skip])

        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return pool, conv

    def dec_unit(self, convh, convl, num_filter, stage):
        if self.dyn_up_sampling:
            up = UpSample()([convh, convl])
        else:
            up = UpSampling2D()(convh)
        up = Convolution2D(num_filter, 2, padding='same', kernel_initializer='he_normal',
                           name="unet_up{}0".format(stage))(up)
        up = Activation(activations.relu)(up)
        merge = Concatenate(axis=3)([convl, up])
        conv = Convolution2D(num_filter, 3, padding='same', kernel_initializer='he_normal',
                             name="unet_up{}1".format(stage))(merge)
        conv = Activation(activations.relu)(conv)
        conv = BatchNormalization()(conv)
        conv = Convolution2D(num_filter, 3, padding='same', kernel_initializer='he_normal',
                             name="unet_up{}2".format(stage))(conv)
        conv = Activation(activations.relu)(conv)
        conv = BatchNormalization()(conv)
        return conv

    def build(self, input_tensor):
        # pool0 = Convolution2D(8, (7, 7), padding="same")(input_tensor)
        input_tensor = Convolution2D(64, (3, 3), padding="same")(input_tensor)
        pool1, conv1 = self.enc_unit(input_tensor, num_filter=(32, 64), stage=1)
        pool2, conv2 = self.enc_unit(pool1, num_filter=(32, 64), stage=2)
        pool3, conv3 = self.enc_unit(pool2, num_filter=(32, 64), stage=3)
        pool4, conv4 = self.enc_unit(pool3, num_filter=(32, 64), stage=4)

        pool5, conv5 = self.enc_unit(pool4, num_filter=(32, 64), stage=5)
        conv6 = self.dec_unit(conv5, conv4, 512, stage=6)

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
