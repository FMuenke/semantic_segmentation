from keras.layers import (
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
    Concatenate,
    Activation
)

from semantic_segmentation.convolutional_neural_network.layer.up_sample import UpSample


class UNet:
    def __init__(self, num_classes, output_function="sigmoid"):
        self.num_classes = num_classes
        self.out_f = output_function

    def build(self, input_tensor):
        conv1 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv11")(input_tensor)
        conv1 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv12")(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv21")(pool1)
        conv2 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv22")(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv31")(pool2)
        conv3 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv32")(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv41")(pool3)
        conv4 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv42")(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Convolution2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv51")(pool4)
        conv5 = Convolution2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv52")(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Convolution2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_up6")(
            UpSampling2D(size=(2, 2))(drop5))
        up6 = UpSample()([up6, drop4])
        merge6 = Concatenate(axis=3)([drop4, up6])
        conv6 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv61")(merge6)
        conv6 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv62")(conv6)

        up7 = Convolution2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_up7")(
            UpSampling2D(size=(2, 2))(conv6))
        up7 = UpSample()([up7, conv3])
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv71")(merge7)
        conv7 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv72")(conv7)

        up8 = Convolution2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_up8")(
            UpSampling2D(size=(2, 2))(conv7))
        up8 = UpSample()([up8, conv2])
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv81")(merge8)
        conv8 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv82")(conv8)

        up9 = Convolution2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_up9")(
            UpSampling2D(size=(2, 2))(conv8))
        up9 = UpSample()([up9, conv1])
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv91")(merge9)
        conv9 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="unet_conv92")(conv9)

        out = Convolution2D(self.num_classes,
                            3,
                            activation=self.out_f,
                            padding='same',
                            kernel_initializer='uniform',
                            name="unet_classification_layer")(conv9)
        return out
