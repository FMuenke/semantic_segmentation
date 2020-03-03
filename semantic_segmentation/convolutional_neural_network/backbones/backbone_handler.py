from semantic_segmentation.convolutional_neural_network.backbones.unet import UNet
from semantic_segmentation.convolutional_neural_network.backbones.pspnet import PSPNet
from semantic_segmentation.convolutional_neural_network.backbones.segnet import SegNet
from semantic_segmentation.convolutional_neural_network.backbones.fr1dz_net import Fr1dzNet


class BackboneHandler:
    def __init__(self, backbone_type, num_classes, output_func="sigmoid"):
        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.output_func = output_func

    def build(self, input_layer):
        if self.backbone_type in ["unet", "unet-relu"]:
            unet = UNet(self.num_classes, output_function=self.output_func)
            return unet.build(input_layer)

        if self.backbone_type == "unet-leaky-relu":
            unet = UNet(self.num_classes, activation="leaky_relu", output_function=self.output_func)
            return unet.build(input_layer)

        if self.backbone_type == "unet_reduced":
            unet = UNet(self.num_classes, reduced=True, output_function=self.output_func)
            return unet.build(input_layer)

        if self.backbone_type == "pspnet":
            pspnet = PSPNet(self.num_classes)
            return pspnet.build(input_layer)

        if self.backbone_type == "segnet":
            segnet = SegNet(self.num_classes, output_function=self.output_func)
            return segnet.build(input_layer)

        if self.backbone_type == "fr1dz":
            segnet = Fr1dzNet(self.num_classes, output_function=self.output_func)
            return segnet.build(input_layer)

        raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

