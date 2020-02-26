from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Input


class BasicNet:
    def __init__(self, num_classes, output_function="sigmoid"):
        self.num_classes = num_classes
        self.output_function = output_function

    def enc_unit(self, x, num_filter, stage):
        conv = Convolution2D(num_filter, kernel_size=3, padding="same", name=f"segnet_conv_{stage}")(x)
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return pool, conv

    def dec_unit(self, x, num_filter, stage):
        conv = UpSampling2D()(x)
        conv = Convolution2D(num_filter, kernel_size=3, padding="same", name=f"segnet_conv_{stage}")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
        return conv

    def log_unit(self, x, num_filter, stage):
        conv = Convolution2D(num_filter, kernel_size=3, padding="same", name=f"segnet_conv_{stage}")(x)
        conv = BatchNormalization()(conv)
        conv = Activation("relu")(conv)
        return conv

    def base_model(self, input_shape):
        base_input = Input(batch_shape=input_shape)
        pool1, conv1 = self.enc_unit(base_input, 64, stage="1")
        pool2, conv2 = self.enc_unit(pool1, 128, stage="2")
        pool3, conv3 = self.enc_unit(pool2, 256, stage="3")

        conv4 = self.log_unit(pool3, 512, stage="4")
        conv5 = self.log_unit(conv4, 512, stage="5")

        conv6 = self.dec_unit(conv5, 256, stage="6")
        conv7 = self.dec_unit(conv6, 128, stage="7")
        conv8 = self.dec_unit(conv7, 64, stage="8")

        cls = Convolution2D(self.num_classes, (1, 1), padding="valid", activation=self.output_function)(conv8)
        return base_input, cls