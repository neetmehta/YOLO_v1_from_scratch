import torch
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms
from torchvision.transforms import transforms
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
from os.path import join as osp
from PIL import Image
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def non_max_suppression(pred_batch, S=(6,20), C=9, prob_threshold=0.4, iou_threshold=0.5, target=False):
    """
    input:
        pred (tensor): [N, S[0], S[1], C+10]
        prob_thres (float): porbability threshold
        iou_thres (float): iou threshold

    return:
        list(tensors[N x 14])
    """
    N = pred_batch.shape[0]
    batch_after_nms = []
    for i in range(N):
        pred = pred_batch[i].unsqueeze(0)
        pred = cellbox_to_corners(pred,S,C, target=target)
        pred = pred[torch.where(pred[...,9]>prob_threshold)]
        pred_boxes = pred[:,-4:]
        pred_scores = pred[:,9]
        pred_boxes_after_nms = nms(pred_boxes, pred_scores, iou_threshold)
        batch_after_nms.append(pred[pred_boxes_after_nms])

    return batch_after_nms


def mAP_tensor(tensor, C=9, target=False):
    boxes = []
    labels = []
    scores = []
    for i in tensor:
        boxes.append(i[-4:])
        labels.append(torch.argmax(i[:C]))
        scores.append(i[C])

    boxes = torch.stack(boxes)
    labels = torch.tensor(labels)
    scores = torch.tensor(scores)
    if target:
        return dict(boxes=boxes, labels=labels)
    return dict(boxes=boxes, scores=scores, labels=labels)

def save_checkpoint(state_dict, path):
    torch.save(state_dict, path)

def eval(dataloader, model, S=(6,20), C=9):

    model.eval()
    loop = tqdm(dataloader)
    batch_size = dataloader.batch_size
    mean_avg_precision = MeanAveragePrecision()
    pred_list = []
    target_list = []

    for image, target in loop:

        image = image.to(device)
        pred = model(image)
        pred = pred.detach().cpu()
        pred = non_max_suppression(pred, S, C)
        target = non_max_suppression(target, S, C, target=True)

        for i in range(batch_size):
            if pred[i].shape[0]==0:
                continue

            pred_dict = mAP_tensor(pred[i], C)
            if len(target[i])==0:
                continue
            target_dict = mAP_tensor(target[i], C, target=True)
            pred_list.append(pred_dict)
            target_list.append(target_dict)
    model.train()
    if len(pred_list)>0:
        mean_avg_precision.update(pred_list, target_list)
        mean_ap = mean_avg_precision.compute()
        return mean_ap

    else:
        return {'map': 0}

def visualize(model, test_dataset, S=(6,20), C=9):
    """
    input:
        model (Yolo model): Trained yolo model
        image (str or tensor([1,3,H,W])): path of image or image tensor (only one image at a time)

    return:

    """
    idx = random.randint(0, len(test_dataset))
    image = test_dataset[idx][0].unsqueeze(0)
    model = model.cpu()
    model.eval()
    pred = model(image)
    pred = non_max_suppression(pred, prob_threshold=0.7)
    image_with_bb = draw_bb(image.squeeze(0), pred)

    image_with_bb.save('vis.jpg')

def draw_bb(img, target):
    to_pil = transforms.ToPILImage()
    img = (img*255).to(torch.uint8)
    img_with_bb = img
    for i in target:
        box = i[:,-4:]
        img_with_bb = draw_bounding_boxes(img, box)
    return to_pil(img_with_bb)