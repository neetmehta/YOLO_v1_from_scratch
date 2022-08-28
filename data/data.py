import torch
from torch.utils.data import Dataset
import os
from os.path import join as osp
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image
from torchvision.datasets import VOCDetection
import math
import numpy as np
from torchvision.utils import draw_bounding_boxes as draw_bb

to_tensor = transforms.Compose([transforms.ToTensor()])

img_transforms = transforms.Compose([transforms.Resize((384,1248)), transforms.ToTensor()])

label_map = {
                'Car':                 0,
                'Van':                 1,
                'Truck':               2,
                'Pedestrian':          3,
                'Person_sitting':      4,
                'Cyclist':             5,
                'Tram':                6,
                'Misc':                7,
                'DontCare':            8
            }

class KittiDetection2D(Dataset):

    def __init__(self, root, S=(6,20), B=2, C=9, transforms=None):
        super(KittiDetection2D, self).__init__()
        self.image_dir = osp(root, "image_2")
        self.label_dir = osp(root, "label_2")
        self.image_list = os.listdir(osp(root, "image_2"))
        self.label_list = os.listdir(osp(root, "label_2"))
        self.transforms = transforms
        self.h = 384
        self.w = 1248
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        return:
            image (Tensor): Image tensor of size [N, H, W]
            target (Tensor): target tensor of size [S[0], S[1], C+B*5]
        """
        image = Image.open(osp(self.image_dir, self.image_list[index]))
        label_pth = osp(self.label_dir, self.label_list[index])
        target = torch.zeros(self.S[0], self.S[1], self.C+5)
        with open(label_pth, 'r') as f:
            for i in f.readlines():
                obj_class, x1, y1, x2, y2 = self._parse_label(i)
                cell, x, y, w, h = self._convert_label(x1, y1, x2, y2)
                target[cell[0], cell[1]] = self._create_vector(obj_class, x, y, w, h)
            
        if self.transforms:
            image = self.transforms(image)

        return image, target

    def _create_vector(self, obj, x, y, h, w):
        obj_vector = torch.zeros(9)
        obj_vector[obj] = 1
        return torch.cat((obj_vector, torch.tensor([1,x,y,h,w])))

    def _parse_label(self, label):
        label = label.split()
        obj_class = label_map[label[0]]
        x1, y1, x2, y2 = int(float(label[4])), int(float(label[5])), int(float(label[6])), int(float(label[7]))
        return obj_class, x1, y1, x2, y2

    def _convert_label(self, x1, y1, x2, y2):
        x = int(round((x1+x2)/2))/self.w
        y = int(round((y1+y2)/2))/self.h
        h = int(round((y2-y1)))/self.h
        w = int(round((x2-x1)))/self.w

        cell = int(y*self.S[0]), int(x*self.S[1])
        x = x*self.S[1]-cell[1]
        y = y*self.S[0]-cell[0]
        h = self.S[0]*h
        w = self.S[1]*w
        return cell,x,y,w,h


classes = {
"aeroplane"     :0,
"bicycle"	    :1,
"bird"	        :2,
"boat"	        :3,
"bottle"        :4,	
"bus"	        :5,
"car"	        :6,
"cat"	        :7,
"chair"	        :8,
"cow"	        :9,
"diningtable"   :10,	
"dog"           :11,
"horse"	        :12,
"motorbike"     :13,	
"person"	    :14,
"pottedplant"   :15,
"sheep"	        :16,
"sofa"	        :17,
"train"	        :18,
"tvmonitor"     :19
}
color_aug_transforms = transforms.Compose([transforms.ColorJitter(brightness=0.5, hue=0.3),
                                     transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5)),
                                     transforms.RandomAdjustSharpness(sharpness_factor=2),
                                     transforms.RandomAutocontrast(),
                                     transforms.RandomEqualize()])


class VOC(VOCDetection):

    def __init__(self, datadir, S, C, B=2, resize=(448,448), image_set='trainval', transform=to_tensor, download=False, augmentation=True) -> None:
        super(VOC, self).__init__(root=datadir, image_set=image_set, download=download)
        self.S = S
        self.C = C
        self.B = B
        self.transform = transform
        self.augmentation = augmentation
        self.resize = transforms.Compose([transforms.Resize(resize)])

    def __getitem__(self, index: int):
        """
        index: int

        image: torch.tensor (N,C,H,W)
        tgt: torch.tensor (N,S[0],S[1],C+5*B) bbox [x,y,w,h]
        """
        image, target = super().__getitem__(index)
        if self.augmentation:
            image = color_aug_transforms(image)
        image = to_tensor(image)
        c, h, w = image.shape 
        image = self.resize(image)
        object_list = target['annotation']["object"]

        bbox = [(classes[i['name']],i['bndbox']) for i in object_list]
        tgt_bbox = [(i[0],[int(i[1]['xmin']), int(i[1]['ymin']),int(i[1]['xmax']),int(i[1]['ymax'])]) for i in bbox]
        map_bb = []
        map_classes = []
        for i in bbox:
            map_bb.append([int(i[1]['xmin']), int(i[1]['ymin']),int(i[1]['xmax']),int(i[1]['ymax'])])
            map_classes.append(i[0])

        map_bb = torch.FloatTensor(map_bb)
        map_target = {'boxes': map_bb, 'labels':torch.tensor(map_classes, dtype=torch.uint8)}
        tgt = self._xml2yolo(tgt_bbox, h, w)
        # C,Y,X = image.shape
        return image, tgt, map_target

    def _xml2yolo(self, bbox, h, w):
        tgt = torch.zeros((self.S[0], self.S[1], self.C+5))
        cell_h = 1./self.S[0]
        cell_w = 1./self.S[1]
        for class_no, (xmin, ymin, xmax, ymax) in bbox:
            x_grid = (xmax+xmin)/2/w
            y_grid = (ymax+ymin)/2/h
            h_grid = (ymax-ymin)/h
            w_grid = (xmax-xmin)/w

            i = math.ceil(y_grid/cell_h)-1
            j = math.ceil(x_grid/cell_w)-1
            tgt[i,j,class_no] = 1
            tgt[i,j,20] = 1
            tgt[i,j,21] = (x_grid - j*cell_w)/cell_w
            tgt[i,j,22] = (y_grid - i*cell_h)/cell_h
            tgt[i,j,23] = w_grid
            tgt[i,j,24] = h_grid

        return tgt
    @staticmethod
    def _yolo2xml(image, target, S):
        c,h,w = image.shape
        pred_dict = {}
        cell_h = 1./S[0]
        cell_w = 1./S[1]
        j = torch.arange(0,14).repeat(14).reshape((14,14))
        i = j.transpose(1,0)
        x_grid = (target[:,:,21]*cell_w + j*cell_w)*target[:,:,20]
        y_grid = (target[:,:,22]*cell_h + i*cell_h)*target[:,:,20]
        h_grid = target[:,:,24]
        w_grid = target[:,:,23]
        xmax = (2*w*x_grid + w_grid*w)/2
        xmin = (2*w*x_grid - w_grid*w)/2
        ymax = (2*h*y_grid + h_grid*h)/2
        ymin = (2*h*y_grid - h_grid*h)/2

        target[:,:,21] = xmin
        target[:,:,22] = ymin
        target[:,:,23] = xmax
        target[:,:,24] = ymax
        bbox = target[target[:,:,20]>0][:,-4:]
        scores = target[target[:,:,20]>0][:,20]
        class_no = target[target[:,:,20]>0][:,:-5]
        classno = torch.argmax(class_no, dim = -1)

        class_no = torch.where(class_no>0)
        class_no = list(class_no[1])
        labels = []
        for key, value in classes.items():
            for i in class_no:
                if i==value:
                    labels.append(key)

        # bbox_after_nms = nms(bbox, scores, iou_threshold=0.5)
        # bbox = bbox[bbox_after_nms,:]
        to_pil = transforms.ToPILImage()
        pil = to_pil(image)
        image = torch.from_numpy(np.asarray(pil))
        image = image.permute(2,0,1)
        image = draw_bb(image, bbox, width=4, font_size=500)
        pred_dict['boxes'] = bbox
        pred_dict['scores'] = scores
        pred_dict['labels'] = classno
        return image, pred_dict



    @staticmethod
    def vis_pred(image, pred, S, threshold=0.5):
        class_no = pred[:,:,:20]
        bb1 = pred[:,:,20:21]
        bb2 = pred[:,:,25:26]
        box1 = pred[:,:,21:25]
        box2 = pred[:,:,26:30]
        bb = torch.cat((bb1,bb2), dim=-1)
        probs ,bb = torch.max(bb, dim=-1)
        bb.unsqueeze_(-1)
        probs.unsqueeze_(-1)
        bb = box2*bb + box1*(1-bb)
        probs[probs<threshold]=0
        probs[probs>=threshold]=1
        target = torch.cat((class_no, probs, bb), dim=-1)
        image, pred_dict = VOC._yolo2xml(image, target, S)
        return image, pred_dict

    