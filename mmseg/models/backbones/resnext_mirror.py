import torch
from torch import nn
from ..builder import BACKBONES

# from backbone.resnext import resnext_101_32x4d_
from .resnext_101_32x4d import resnext


@BACKBONES.register_module('ResNeXt_Mirror')
class ResNeXt101(nn.Module):
    def __init__(self, backbone_path):
        super(ResNeXt101, self).__init__()
        # net = resnext_101_32x4d_.resnext_101_32x4d
        net = resnext
        if backbone_path is not None:
            weights = torch.load(backbone_path)
            net.load_state_dict(weights, strict=True)
            print("Load ResNeXt Weights Succeed!")

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        x_list = []

        layer0 = self.layer0(x)
        x_list.append(layer0)

        layer1 = self.layer1(layer0)
        x_list.append(layer1)

        layer2 = self.layer2(layer1)
        x_list.append(layer2)

        layer3 = self.layer3(layer2)
        x_list.append(layer3)

        layer4 = self.layer4(layer3)
        x_list.append(layer4)
        return x_list
