import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
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

def cellbox_to_corners(pred, S, C, target=False):
    """
    input:
        pred (tensor): [N, S[0], S[1], C+5*B]
        S (tuple): Cell division of
        C (int): No. of classes

    output:
        (tensor): Output tensor with last 4 values as x1,y1,x2,y2 i.e. corner of bounding box
    """
    if target:
        pred = torch.cat((pred, pred[...,-5:]), dim=-1)
    N = pred.shape[0]
    max_prob, bestbox = torch.max(torch.cat((pred[..., C].unsqueeze(0), pred[..., C+5].unsqueeze(0)), dim=0), dim=0)
    bestbox.unsqueeze_(-1)
    pred_box = (1-bestbox)*pred[..., C+1:C+5] + bestbox*pred[..., C+6:C+10]
    pred_prob = (1-bestbox)*pred[..., C:C+1] + bestbox*pred[..., C+5:C+6]
    cell_y = torch.arange(S[0]).repeat(N,S[1],1).unsqueeze(-1)
    cell_y = torch.transpose(cell_y, 1, 2)
    cell_x = torch.arange(S[1]).repeat(N,S[0],1).unsqueeze(-1)
    x = (1/S[1])*(pred_box[...,0:1]+cell_x)*1248
    y = (1/S[0])*(pred_box[...,1:2]+cell_y)*384
    w = (1/S[1])*pred_box[...,2:3]*1248
    h = (1/S[0])*pred_box[...,3:4]*384

    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2

    return torch.cat((pred[...,:C],pred_prob, x1,y1,x2,y2), dim=-1)

def cellbox_to_imgbox(pred, S, C, target=False):
    """
    input:
        pred (tensor): [N, S[0], S[1], C+5*B]
        S (tuple): Cell division of
        C (int): No. of classes

    output:
        (tensor): Output tensor relative to image rather than cell

    """
    if target:
        pred = torch.cat((pred, pred[...,-5:]), dim=-1)
    N = pred.shape[0]
    max_prob, bestbox = torch.max(torch.cat((pred[..., C].unsqueeze(0), pred[..., C+5].unsqueeze(0)), dim=0), dim=0)
    bestbox.unsqueeze_(-1)
    pred_box = (1-bestbox)*pred[..., C+1:C+5] + bestbox*pred[..., C+6:C+10]
    pred_prob = (1-bestbox)*pred[..., C:C+1] + bestbox*pred[..., C+5:C+6]
    cell_y = torch.arange(S[0]).repeat(N,S[1],1).unsqueeze(-1)
    cell_y = torch.transpose(cell_y, 1, 2)
    cell_x = torch.arange(S[1]).repeat(N,S[0],1).unsqueeze(-1)
    x = (1/S[1])*(pred_box[...,0:1]+cell_x)
    y = (1/S[0])*(pred_box[...,1:2]+cell_y)
    w = (1/S[1])*pred_box[...,2:3]
    h = (1/S[0])*pred_box[...,3:4]

    return torch.cat((pred[...,:C],pred_prob, x,y,w,h), dim=-1)
def iou_check(pred, target, thres):
    pred = pred.reshape(1,1,1,-1)
    target = target.reshape(1,1,1,-1)
    return iou(pred, target)<thres
def non_max_suppression(pred, S=(6,20), C=9, prob_threshold=0.4, iou_threshold=0.5):
    """
    input:
        pred (tensor): [N, S[0], S[1], C+10]
        prob_thres (float): porbability threshold
        iou_thres (float): iou threshold

    return:
        list(Nxlist(tensors[14])
    """
    all_images_bb_after_nms = []
    for i in range(pred.shape[0]):
        pred = cellbox_to_imgbox(pred, S, C)
        bb_after_nms = []
        bbox_list = list(pred.reshape(-1,C+5))
        bboxes = [bb for bb in bbox_list if bb[-5]>prob_threshold]
        bboxes = sorted(bboxes, key=lambda x:x[-5], reverse=True)
        while bboxes:
            chosen_box = bboxes.pop(0)

            bboxes = [bb for bb in bboxes if torch.argmax(bb[:C])!=torch.argmax(chosen_box[:C]) or iou_check(bb[-4:], chosen_box[-4:], iou_threshold)]
            bb_after_nms.append(chosen_box)
        
        all_images_bb_after_nms.append(bb_after_nms)

    return all_images_bb_after_nms