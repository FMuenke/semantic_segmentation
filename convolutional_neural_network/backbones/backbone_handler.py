from convolutional_neural_network.backbones.unet import UNet


class BackboneHandler:
    def __init__(self, backbone_type, num_classes):
        self.backbone_type = backbone_type
        self.num_classes = num_classes

    def build(self, input_layer):
        if self.backbone_type == "unet":
            unet = UNet(self.num_classes)
            return unet.build(input_layer)

        raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

