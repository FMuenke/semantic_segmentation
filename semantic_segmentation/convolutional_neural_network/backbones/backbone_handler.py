from keras.layers import Dense, Convolution2D, Flatten, AveragePooling2D, Concatenate, Reshape

from semantic_segmentation.convolutional_neural_network.backbones.unet import UNet
from semantic_segmentation.convolutional_neural_network.backbones.pspnet import PSPNet
from semantic_segmentation.convolutional_neural_network.backbones.segnet import SegNet
from semantic_segmentation.convolutional_neural_network.backbones.fr1dz_net import Fr1dzNet
from semantic_segmentation.convolutional_neural_network.backbones.basic_net import BasicNet
from semantic_segmentation.convolutional_neural_network.backbones.dynamic_net import DynamicNet

from semantic_segmentation.convolutional_neural_network.losses import dice, focal_loss, jaccard, weighted_cross_entropy, mixed


class BackboneHandler:
    def __init__(self, backbone_type, num_classes, output_func="sigmoid", loss_type="bc"):
        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.output_func = output_func
        self.loss_type = loss_type

    def loss(self):
        if self.output_func == "ellipse":
            return ["binary_crossentropy", "mae"]

        if self.loss_type in ["focal", "focal_loss"]:
            return focal_loss()
        if self.loss_type in ["binary_crossentropy", "bc"]:
            return "binary_crossentropy"
        if self.loss_type == "dice":
            return dice()
        if self.loss_type == "jaccard":
            return jaccard()
        if self.loss_type == "weighted_cross_entropy":
            return weighted_cross_entropy(100)
        if self.loss_type == "mixed":
            return mixed()
        if self.loss_type in ["mean_squared_error", "mse"]:
            return "mean_squared_error"

        raise ValueError("Loss Type unknown {}".format(self.loss_type))

    def metric(self):
        if self.output_func == "ellipse":
            return ["accuracy", "mae"]

        return ["accuracy"]

    def _build(self, input_layer, num_classes, output_func):
        if self.backbone_type in ["unet", "unet-relu"]:
            model = UNet(num_classes, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type == "unet-leaky-relu":
            model = UNet(num_classes, activation="leaky_relu", output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type == "unet_reduced":
            model = UNet(num_classes, reduced=True, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type == "pspnet":
            model = PSPNet(num_classes)
            return model.build(input_layer)

        if self.backbone_type == "segnet":
            model = SegNet(num_classes, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type == "fr1dz":
            model = Fr1dzNet(num_classes, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type == "basic_net":
            model = BasicNet(num_classes, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type == "dynamic_net":
            model = DynamicNet(num_classes, output_function=output_func)
            return model.build(input_layer)

        raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

    def build_ellipse_net(self, input_layer):
        x = self._build(input_layer, self.num_classes, "sigmoid")

        e = Convolution2D(8, (8, 8), strides=(2, 2), activation="relu")(x)
        e = AveragePooling2D((2, 2))(e)
        e = Convolution2D(16, (4, 4), strides=(2, 2), activation="relu")(e)
        e = AveragePooling2D((2, 2))(e)
        e = Flatten()(e)
        e = Dense(5)(e)

        return [x, e]

    def build(self, input_layer):
        if self.output_func == "ellipse":
            return self.build_ellipse_net(input_layer)

        return self._build(input_layer, self.num_classes, self.output_func)

