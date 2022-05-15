import torch
from torch import nn
from torchvision.models import resnet152, resnet101, resnet50, resnet34, resnet18

class Darknet(nn.Module):
    def __init__(self, model_cfg, in_channels=3, fcl_out=496, S=(6,20), B=2, num_classes=9, dropout=0.06):
        super(Darknet, self).__init__()
        layer_list = []
        for block in model_cfg['architecture']:
            for layer in block:
                if isinstance(layer, list):
                    layer_list.append(self._create_conv_layer(in_channels, layer[0], layer[1], layer[2], layer[3]))
                    in_channels = layer[0]
                elif layer == 'M':
                    layer_list.append(nn.MaxPool2d(2,2))

                else:
                    raise Exception('Invalid layer')
        self.S = S
        self.B = B
        self.C = num_classes
        if isinstance(S, int):
            self.S = (S,S)

        self.backbone = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.backbone(x)
        return x

    def _create_conv_layer(self, in_channels: int, out_channels: int, kernal: int, stride: int, padding: int):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernal, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.1))


class ResNet(nn.Module):

    def __init__(self, model_type, pretrained=True) -> None:
        super(ResNet, self).__init__()
        if model_type == 'resnet152':
            self.resnet = resnet152(pretrained)

        if model_type == 'resnet101':
            self.resnet = resnet101(pretrained)

        if model_type == 'resnet50':
            self.resnet = resnet50(pretrained)

        if model_type == 'resnet34':
            self.resnet = resnet34(pretrained)

        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained)

        del self.resnet._modules['fc']

    def forward(self, x):
        x = self.resnet._modules['conv1'](x)
        x = self.resnet._modules['bn1'](x)
        x = self.resnet._modules['maxpool'](x)
        x = self.resnet._modules['layer1'](x)
        x = self.resnet._modules['layer2'](x)
        x = self.resnet._modules['layer3'](x)
        x = self.resnet._modules['layer4'](x)
        return x
