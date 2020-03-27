from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

if K.backend() == "tensorflow":
    import tensorflow as tf


class CroppingLayer(Layer):
    def __init__(self, crop_size, crop_coordinates, **kwargs):

        self.crop_size = crop_size
        self.crop_coordinates = crop_coordinates

        super(CroppingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[3]

    def compute_output_shape(self, input_shape):
        return None, self.crop_size, self.crop_size, self.nb_channels

    def call(self, x, mask=None):
        batch_size, height, width, nb_features = x.get_shape().as_list()

        x1 = self.crop_coordinates[0] * width
        y1 = self.crop_coordinates[1] * height
        x2 = self.crop_coordinates[2] * width
        y2 = self.crop_coordinates[3] * height

        if x2 - x1 < 1:
            x2 = x1 + 1
        if y2 - y1 < 1:
            y2 = y1 + 1

        x1 = K.cast(x1, "int32")
        y1 = K.cast(y1, "int32")
        x2 = K.cast(x2, "int32")
        y2 = K.cast(y2, "int32")

        rs = tf.image.resize(
            x[:, y1:y2, x1:x2, :], (self.crop_size, self.crop_size)
        )

        final_output = K.reshape(
            rs,
            (-1, self.crop_size, self.crop_size, self.nb_channels),
        )

        return final_output

    def get_config(self):
        config = {"crop_size": self.crop_size, "crop_coordinates": self.crop_coordinates}
        base_config = super(CroppingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
