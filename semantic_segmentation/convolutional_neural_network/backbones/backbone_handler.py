from semantic_segmentation.convolutional_neural_network.backbones.unet import UNet
from semantic_segmentation.convolutional_neural_network.backbones.pspnet import PSPNet
from semantic_segmentation.convolutional_neural_network.backbones.segnet import SegNet
from semantic_segmentation.convolutional_neural_network.backbones.fr1dz_net import Fr1dzNet
from semantic_segmentation.convolutional_neural_network.backbones.basic_net import BasicNet
from semantic_segmentation.convolutional_neural_network.backbones.dynamic_net import DynamicNet


class BackboneHandler:
    def __init__(self, backbone_type, num_classes, output_func="sigmoid"):
        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.output_func = output_func

    def build(self, input_layer):
        if self.backbone_type in ["unet", "unet-relu"]:
            model = UNet(self.num_classes, output_function=self.output_func)
            return model.build(input_layer)

        if self.backbone_type == "unet-leaky-relu":
            model = UNet(self.num_classes, activation="leaky_relu", output_function=self.output_func)
            return model.build(input_layer)

        if self.backbone_type == "unet_reduced":
            model = UNet(self.num_classes, reduced=True, output_function=self.output_func)
            return model.build(input_layer)

        if self.backbone_type == "pspnet":
            model = PSPNet(self.num_classes)
            return model.build(input_layer)

        if self.backbone_type == "segnet":
            model = SegNet(self.num_classes, output_function=self.output_func)
            return model.build(input_layer)

        if self.backbone_type == "fr1dz":
            model = Fr1dzNet(self.num_classes, output_function=self.output_func)
            return model.build(input_layer)

        if self.backbone_type == "basic_net":
            model = BasicNet(self.num_classes, output_function=self.output_func)
            return model.build(input_layer)

        if self.backbone_type == "dynamic_net":
            model = DynamicNet(self.num_classes, output_function=self.output_func)
            return model.build(input_layer)

        raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

