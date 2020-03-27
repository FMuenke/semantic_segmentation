from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class GlobalSumLayer(Layer):
    def __init__(self, **kwargs):
        super(GlobalSumLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]

    def call(self, x, mask=None):
        x = K.sum(x, axis=2)
        x = K.sum(x, axis=1)
        return x

    def get_config(self):
        base_config = super(GlobalSumLayer, self).get_config()
        return dict(list(base_config.items()))
