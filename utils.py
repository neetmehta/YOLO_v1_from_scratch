import torch
import yaml

def iou(pred, tar):
    """
    input:
        pred (Tensor): [N, S[0], S[1], 4] -> [N,11,24,4]
        tar (Tensor): [N, S[0], S[1], 4] -> [N,11,24,4]

    return:
        iou (Tensor): [N, S[0], S[1]]
    """
    
    pred_box = torch.cat((pred[:,:,:,0:1]-pred[:,:,:,2:3]/2, 
                          pred[:,:,:,1:2]-pred[:,:,:,3:4]/2, 
                          pred[:,:,:,0:1]+pred[:,:,:,2:3]/2, 
                          pred[:,:,:,1:2]+pred[:,:,:,3:4]/2), dim=-1)

    tar_box = torch.cat((tar[:,:,:,0:1]-tar[:,:,:,2:3]/2, 
                        tar[:,:,:,1:2]-tar[:,:,:,3:4]/2, 
                        tar[:,:,:,0:1]+tar[:,:,:,2:3]/2, 
                        tar[:,:,:,1:2]+tar[:,:,:,3:4]/2), dim=-1)

    x_top_left = torch.max(pred_box[:,:,:,0], tar_box[:,:,:,0])
    y_top_left = torch.max(pred_box[:,:,:,1], tar_box[:,:,:,1])
    x_bottom_right = torch.min(pred_box[:,:,:,2], tar_box[:,:,:,2])
    y_bottom_right = torch.min(pred_box[:,:,:,3], tar_box[:,:,:,3])    

    intersection = torch.clamp((x_bottom_right-x_top_left),0)*torch.clamp((y_bottom_right-y_top_left),0)
    pred_box_area = (pred_box[:,:,:,2]-pred_box[:,:,:,0])*(pred_box[:,:,:,3]-pred_box[:,:,:,1])
    tar_box_area = (tar_box[:,:,:,2]-tar_box[:,:,:,0])*(tar_box[:,:,:,3]-tar_box[:,:,:,1])
    union = pred_box_area + tar_box_area - intersection    

    return intersection/union

def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)