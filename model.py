import torch
from torch import nn
import yaml
from encoder import Darknet, ResNet
from utils import read_yaml

class YOLOv1(nn.Module):
    def __init__(self, backbone, in_channels=3, fcl_out=496, S=(6,20), B=2, C=9, backbone_out_channels=1024, dropout=0.06):
        super(YOLOv1, self).__init__()
        self.backbone = backbone
        self.S = S
        self.B = B
        self.C = C
        if isinstance(S, int):
            self.S = (S,S)

        self.fcl = self._create_fcl(self.S, self.B, self.C, fcl_out, backbone_out_channels, dropout)

    def forward(self, x):
        x = self.backbone(x)
        flat = nn.Flatten()
        x = flat(x)
        x = self.fcl(x)
        x = x.reshape(-1, self.S[0], self.S[1], self.C + self.B*5)
        return x

    def _create_fcl(self, S: tuple, B: int, C: int, fcl_out: int=4096, backbone_out_channels=1024, dropout=0.0):
        return nn.Sequential(nn.Flatten(), nn.Linear(backbone_out_channels*S[0]*S[1], fcl_out), nn.Dropout(dropout), nn.LeakyReLU(0.1), nn.Linear(fcl_out, S[0]*S[1]*(self.C + self.B*5)))

def get_model(backbone: str, S, C, fcl_out: int=496, dropout=0.0):
    if backbone == 'darknet':
        darknet_cfg = read_yaml('yaml/darknet.yaml')
        backbone_model = Darknet(darknet_cfg)
        model = YOLOv1(backbone=backbone_model, S=S, C=C, backbone_out_channels=1024, fcl_out=fcl_out, dropout=dropout)

    if backbone == 'resnet152':
        backbone_model = ResNet(backbone)
        model = YOLOv1(backbone=backbone_model, S=S, C=C, backbone_out_channels=2048, fcl_out=fcl_out, dropout=dropout)

    if backbone == 'resnet101':
        backbone_model = ResNet(backbone)
        model = YOLOv1(backbone=backbone_model, S=S, C=C, backbone_out_channels=2048, fcl_out=fcl_out, dropout=dropout)

    if backbone == 'resnet50':
        backbone_model = ResNet(backbone)
        model = YOLOv1(backbone=backbone_model, S=S, C=C, backbone_out_channels=2048, fcl_out=fcl_out, dropout=dropout)

    if backbone == 'resnet34':
        backbone_model = ResNet(backbone)
        model = YOLOv1(backbone=backbone_model, S=S, C=C, backbone_out_channels=512, fcl_out=fcl_out, dropout=dropout)

    if backbone == 'resnet18':
        backbone_model = ResNet(backbone)
        model = YOLOv1(backbone=backbone_model, S=S, C=C, backbone_out_channels=512, fcl_out=fcl_out, dropout=dropout)

    return model