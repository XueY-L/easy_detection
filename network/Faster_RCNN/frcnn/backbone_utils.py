from collections import OrderedDict
import torch
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet


class BackboneWithFPN(nn.Sequential):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        super(BackboneWithFPN, self).__init__(OrderedDict(
            [("body", body), ("fpn", fpn)]))  # body还是resnet，IntermediateLayerGetter返回forward过程中return_layers层的输出，我猜用作FeaturePyramidNetwork的输入；FeaturePyramidNetwork接收resnet的中间输出，输出字典，每个键值对代表不同尺度的特征图
        self.out_channels = out_channels


def resnet_fpn_backbone(backbone_name, pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d):
    backbone = resnet.__dict__[backbone_name](  # 就是个resnet
        pretrained=pretrained,
        #norm_layer=torch.nn.SyncBatchNorm)
        norm_layer=norm_layer)

    # freeze layers
    for name, parameter in backbone.named_parameters():  # 为什么冻结layer2前面的？
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

    in_channels_stage2 = backbone.inplanes // 8  # backbone.inplanes是fc的输入维度
    in_channels_list = [  # 以res50为例，是四个layer的输出维度
        in_channels_stage2,  # 256
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256

    out = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    return out
