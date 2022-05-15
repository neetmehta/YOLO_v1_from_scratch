import torch
from torch import nn
from utils import iou

class YoloLoss(nn.Module):
    def __init__(self, S=(6, 20), B=2, C=9, coord=5, noobj=0.5) -> None:
        super(YoloLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.coord = coord
        self.noobj = noobj

    def forward(self, pred, target):
        C = self.C
        class_probs = target[..., :C]                              # [N, S[0], s[1], C]
        exist_box_identity = target[..., C:C+1]
        not_exist_box_identity = 1-exist_box_identity               # [N, S[0], s[1], 1]
        target_box = exist_box_identity * target[..., C+1:]                       # [N, S[0], s[1], 4]
        
        iou_b1 = iou(pred[..., C+1:C+5], target[...,C+1:C+5]).unsqueeze(-1) # [N, S[0], S[1], 1]
        iou_b2 = iou(pred[...,C+6:C+10], target[..., C+1:C+5]).unsqueeze(-1) # [N, S[0], S[1], 1]
        max_iou, best_box = torch.max(torch.cat((iou_b1,iou_b2), dim=-1), dim=-1)            # best_box [N, S[0], S[1]]
        best_box = best_box.unsqueeze(-1)
        
        pred_best_box = exist_box_identity*(best_box*pred[...,C+6:C+10] + (1-best_box)*pred[..., C+1:C+5]) # [N, S[0], s[1], 4]
        
        ## coord loss
        pred_coord = pred_best_box[..., 0:2]
        target_coord = target_box[..., 0:2]
        coord_loss = self.mse_loss(torch.flatten(pred_coord, 0, -2), torch.flatten(target_coord, 0, -2))
        

        ## box loss
        
        pred_h_w = torch.sign(pred_best_box[...,2:4])*torch.sqrt(torch.abs(pred_best_box[..., 2:4] + 1e-6)) # [N, S[0], S[1], 2]
        target_h_w = torch.sqrt(target_box[..., 2:4]) #[N, S[0], S[1], 2]
        box_loss = self.mse_loss(torch.flatten(pred_h_w, 0, -2), torch.flatten(target_h_w, 0, -2))
        
        ## object loss
        pred_obj = exist_box_identity*(best_box*pred[...,C+5:C+6] + (1-best_box)*pred[..., C:C+1])

        object_loss = self.mse_loss(torch.flatten(pred_obj, 0, -2), torch.flatten(exist_box_identity, 0, -2))
        
        ## no object loss

        noobject_loss = self.mse_loss(torch.flatten((1-exist_box_identity)*pred[...,C:C+1], start_dim=1), torch.flatten((1-exist_box_identity)*target[...,C:C+1], start_dim=1)) + self.mse_loss(torch.flatten((1-exist_box_identity)*pred[...,C+5:C+6], start_dim=1), torch.flatten((1-exist_box_identity)*target[...,C:C+1], start_dim=1))
        
        ## class loss
        pred_class = exist_box_identity*pred[..., :C]
        target_class = exist_box_identity*target[..., :C]

        class_loss = self.mse_loss(torch.flatten(pred_class, 0, -2), torch.flatten(target_class, 0, -2))

        # print('cord loss', coord_loss)
        # print('box loss', box_loss)
        # print('obj loss', object_loss)
        # print('noobj loss', noobject_loss)
        # print('class loss', class_loss)
        
        return self.coord*box_loss + self.coord*coord_loss + object_loss + self.noobj*noobject_loss + class_loss
        