from keras.layers import (
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
    Dropout,
    Concatenate,
    Activation
)

from convolutional_neural_network.layer.up_sample import UpSample


class UNet:
    def __init__(self, num_classes, output_function="sigmoid"):
        self.num_classes = num_classes
        self.out_f = output_function

    def build(self, input_tensor):
        conv1 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_tensor)
        conv1 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Convolution2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Convolution2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Convolution2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        up6 = UpSample()([up6, drop4])
        merge6 = Concatenate(axis=3)([drop4, up6])
        conv6 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Convolution2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Convolution2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        up7 = UpSample()([up7, conv3])
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Convolution2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        up8 = UpSample()([up8, conv2])
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Convolution2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        up9 = UpSample()([up9, conv1])
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        out = Convolution2D(self.num_classes,
                            3,
                            activation=self.out_f,
                            padding='same',
                            kernel_initializer='uniform')(conv9)
        return out
