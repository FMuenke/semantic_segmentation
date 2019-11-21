from keras.layers import Layer
import tensorflow as tf


class UpSample(Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor. First Tesnor is shape to size of second one
    """
    def __init__(self, **kwargs):
        super(UpSample, self).__init__(**kwargs)

    def call(self, x, mask=None):
        x_in, target = x
        batch_size, height, width, nb_features = target.get_shape().as_list()
        return tf.image.resize_images(x_in, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
