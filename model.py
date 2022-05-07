import torch
from torch import nn

class YOLOv1(nn.Module):
    def __init__(self, model_cfg, in_channels=3, fcl_out=4096, S=(11,24), B=2, num_classes=9):
        super(YOLOv1, self).__init__()
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
        self.fcl = self._create_fcl(self.S, self.B, self.C, fcl_out)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fcl(x)
        x = x.reshape(-1, self.S[0], self.S[1], self.C + self.B*5)
        return x

    def _create_conv_layer(self, in_channels: int, out_channels: int, kernal: int, stride: int, padding: int):
        return nn.Conv2d(in_channels, out_channels, kernal, stride, padding)

    def _create_fcl(self, S: tuple, B: int, C: int, fcl_out: int=4096):
        return nn.Sequential(nn.Flatten(), nn.Linear(1024*S[0]*S[1], fcl_out), nn.Dropout(0.0), nn.LeakyReLU(0.1), nn.Linear(4096, S[0]*S[1]*(self.C + self.B*5)))

