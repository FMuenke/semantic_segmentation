from tensorflow.keras.layers import (
    Activation,
    Convolution2D,
    BatchNormalization,
    Add,
)


def conv_block(
    input_tensor, kernel_size, filters, stage, block, strides=(2, 2)
):
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = "f1_res" + str(stage) + block + "_branch"

    x = Convolution2D(
        nb_filter1,
        (1, 1),
        strides=strides,
        name=conv_name_base + "2a",
        kernel_initializer="he_normal",
    )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter3, (1, 1), name=conv_name_base + "2c"
    )(x)
    x = BatchNormalization()(x)

    shortcut = Convolution2D(
        nb_filter3,
        (1, 1),
        strides=strides,
        name=conv_name_base + "1",
        kernel_initializer="he_normal",
    )(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters

    conv_name_base = "f1_res" + str(stage) + block + "_branch"

    x = Convolution2D(
        nb_filter1, (1, 1), name=conv_name_base + "2a",
    )(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter2,
        (kernel_size, kernel_size),
        padding="same",
        name=conv_name_base + "2b",
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Convolution2D(
        nb_filter3, (1, 1), name=conv_name_base + "2c", kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)

    x = Add()([x, input_tensor])
    x = Activation("relu")(x)
    return x
