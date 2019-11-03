from convolutional_neural_network.resnet import ResNet


class BackboneHandler:
    def __init__(self, backbone_type):
        self.backbone_type = backbone_type

    def build(self, input_layer):
        if self.backbone_type == "resnet":
            resnet = ResNet(trainable=True)
            return resnet.build_backbone(input_layer)

        raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

