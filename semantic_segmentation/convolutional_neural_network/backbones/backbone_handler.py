from semantic_segmentation.convolutional_neural_network.backbones.unet import UNet
from semantic_segmentation.convolutional_neural_network.backbones.pspnet import PSPNet
from semantic_segmentation.convolutional_neural_network.backbones.segnet import SegNet
from semantic_segmentation.convolutional_neural_network.backbones.fr1dz_net import Fr1dzNet


class BackboneHandler:
    def __init__(self, backbone_type, num_classes):
        self.backbone_type = backbone_type
        self.num_classes = num_classes

    def build(self, input_layer):
        if self.backbone_type == "unet":
            unet = UNet(self.num_classes)
            return unet.build(input_layer)

        if self.backbone_type == "pspnet":
            pspnet = PSPNet(self.num_classes)
            return pspnet.build(input_layer)

        if self.backbone_type == "segnet":
            segnet = SegNet(self.num_classes)
            return segnet.build(input_layer)

        if self.backbone_type == "fr1dz":
            segnet = Fr1dzNet(self.num_classes)
            return segnet.build(input_layer)

        raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

