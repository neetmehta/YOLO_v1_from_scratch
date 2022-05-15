import torch
from torch import nn
from torchvision.models import resnet101

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


class ResNet101(nn.Module):

    def __init__(self, pretrained=True) -> None:
        super(ResNet101, self).__init__()
        self.resnet101 = resnet101(pretrained)
        del self.resnet101._modules['fc']

    def forward(self, x):
        x = self.resnet101._modules['conv1'](x)
        x = self.resnet101._modules['bn1'](x)
        x = self.resnet101._modules['maxpool'](x)
        x = self.resnet101._modules['layer1'](x)
        x = self.resnet101._modules['layer2'](x)
        x = self.resnet101._modules['layer3'](x)
        x = self.resnet101._modules['layer4'](x)
        return x