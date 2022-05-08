import torch
from torch import nn
from utils import iou

class YoloLoss(nn.Module):
    def __init__(self, S=(11,24), B=2, C=9, coord=5, noobj=0.5) -> None:
        super(YoloLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.S = S
        self.B = B
        self.C = C
        self.coord = coord
        self.noobj = noobj

    def forward(self, pred, target):

        class_probs = target[..., :self.C]                              # [N, S[0], s[1], C]
        exist_box_identity = target[..., self.C:self.C+1]
        not_exist_box_identity = 1-exist_box_identity               # [N, S[0], s[1], 1]
        target_box = exist_box_identity * target[..., self.C+1:]                       # [N, S[0], s[1], 4]
        
        iou_b1 = iou(pred[..., 10:14], target[...,10:14]).unsqueeze(-1) # [N, S[0], S[1], 1]
        iou_b2 = iou(pred[...,15:19], target[..., 10:14]).unsqueeze(-1) # [N, S[0], S[1], 1]
        max_iou, best_box = torch.max(torch.cat((iou_b1,iou_b2), dim=-1), dim=-1)            # best_box [N, S[0], S[1]]
        best_box = best_box.unsqueeze(-1)
        pred_best_box = exist_box_identity*(best_box*pred[...,15:19] + (1-best_box)*pred[..., 10:14]) # [N, S[0], s[1], 4]
        
        ## coord loss
        pred_coord = pred_best_box[..., 0:2]
        target_coord = target_box[..., 0:2]
        coord_loss = self.coord*self.mse_loss(torch.flatten(pred_best_box, 0, -2), torch.flatten(target_box, 0, -2))

        ## box loss
        
        pred_h_w = torch.sign(pred_best_box[...,2:4])*torch.sqrt(torch.abs(pred_best_box[..., 2:4])) # [N, S[0], S[1], 2]
        target_h_w = torch.sqrt(target_box[..., 2:4]) #[N, S[0], S[1], 2]
        box_loss = self.coord*self.mse_loss(torch.flatten(pred_h_w, 0, -2), torch.flatten(target_h_w, 0, -2))

        ## object loss
        pred_obj = exist_box_identity*(best_box*pred[...,14:15] + (1-best_box)*pred[..., 9:10])

        object_loss = self.mse_loss(torch.flatten(pred_obj, 0, -2), torch.flatten(exist_box_identity, 0, -2))

        ## no object loss

        noobject_loss = self.mse_loss(torch.flatten((1-exist_box_identity)*pred[...,9:10], 0, -2), torch.flatten((1-exist_box_identity)*target[...,9:10], 0, -2)) + self.mse_loss(torch.flatten((1-exist_box_identity)*pred[...,9:10], 0, -2), torch.flatten((1-exist_box_identity)*target[...,9:10], 0, -2))      ###### CHECK
        noobject_loss = self.noobj*object_loss

        ## class loss
        pred_class = exist_box_identity*pred[..., :9]
        target_class = exist_box_identity*target[..., :9]

        class_loss = self.mse_loss(torch.flatten(pred_class, 0, -2), torch.flatten(target_class, 0, -2))

        return box_loss + coord_loss + object_loss + noobject_loss + class_loss
        
