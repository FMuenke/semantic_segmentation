from keras.layers import (
    Activation,
    BatchNormalization,
    Convolution2D,
    MaxPooling2D,
)

from semantic_segmentation.convolutional_neural_network.layer.up_sample import UpSample


class SegNet:
    def __init__(self, num_classes, output_function="sigmoid"):
        self.num_classes = num_classes
        self.out_f = output_function

    def build(self, input_tensor):
        x = input_tensor
        # Encoder
        x = Convolution2D(64, kernel_size=3, padding="same", name="segnet_conv1")(x)
        x = BatchNormalization()(x)
        st_1 = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(st_1)

        x = Convolution2D(128, kernel_size=3, padding="same", name="segnet_conv2")(x)
        x = BatchNormalization()(x)
        st_2 = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(st_2)

        x = Convolution2D(256, kernel_size=3, padding="same", name="segnet_conv3")(x)
        x = BatchNormalization()(x)
        st_3 = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(st_3)

        x = Convolution2D(512, kernel_size=3, padding="same", name="segnet_conv4")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Decoder
        x = Convolution2D(512, kernel_size=3, padding="same", name="segnet_conv5")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSample()([x, st_3])
        x = Convolution2D(256, kernel_size=3, padding="same", name="segnet_conv6")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSample()([x, st_2])
        x = Convolution2D(128, kernel_size=3, padding="same", name="segnet_conv7")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = UpSample()([x, st_1])
        x = Convolution2D(64, kernel_size=3, padding="same", name="segnet_conv8")(x)
        x = BatchNormalization()(x)

        x = Convolution2D(self.num_classes, kernel_size=3, padding="same", name="segnet_classification")(x)
        out = Activation(self.out_f)(x)
        return out
