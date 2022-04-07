from semantic_segmentation.convolutional_neural_network.backbones.unet import UNet
from semantic_segmentation.convolutional_neural_network.backbones.pspnet import PSPNet
from semantic_segmentation.convolutional_neural_network.backbones.segnet import SegNet
from semantic_segmentation.convolutional_neural_network.backbones.deeplabv3 import Deeplabv3
from semantic_segmentation.convolutional_neural_network.backbones.residual_unet import ResidualUNet
from semantic_segmentation.convolutional_neural_network.backbones.unet_plus import UNetPlus

from semantic_segmentation.convolutional_neural_network.losses import dice, focal_loss, jaccard, weighted_cross_entropy, mixed


class BackboneHandler:
    def __init__(self, backbone_type, num_classes, output_func="sigmoid", loss_type="bc"):
        self.backbone_type = backbone_type
        self.num_classes = num_classes
        self.output_func = output_func
        self.loss_type = loss_type

    def loss(self):
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
        return ["accuracy"]

    def build(self, input_layer):
        num_classes = self.num_classes
        output_func = self.output_func
        if self.backbone_type in ["unet", "unet-relu"]:
            model = UNet(num_classes, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type in ["res-unet"]:
            model = ResidualUNet(num_classes, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type in ["unet+", "unet_plus", "unet-plus"]:
            model = UNetPlus(num_classes, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type in ["unet-no-batchnorm"]:
            model = UNet(num_classes, batch_norm=False, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type in ["deeplabv3"]:
            model = Deeplabv3(num_classes, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type == "unet-leaky-relu":
            model = UNet(num_classes, activation="leaky_relu", output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type in ["unet_reduced", "unet_small"]:
            model = UNet(num_classes, reduced=True, output_function=output_func)
            return model.build(input_layer)

        if self.backbone_type == "pspnet":
            model = PSPNet(num_classes)
            return model.build(input_layer)

        if self.backbone_type == "segnet":
            model = SegNet(num_classes, output_function=output_func)
            return model.build(input_layer)

        raise ValueError("{} Backbone was not recognised".format(self.backbone_type))

