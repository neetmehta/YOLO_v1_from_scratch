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
class VOC(VOCDetection):

    def __init__(self, datadir, S, C, B=2, image_set='trainval', transform=to_tensor, download=False) -> None:
        super(VOC, self).__init__(root=datadir, image_set=image_set, download=download)
        self.S = S
        self.C = C
        self.B = B
        self.transform = transform

    def __getitem__(self, index: int):
        """
        index: int

        image: torch.tensor (N,C,H,W)
        tgt: torch.tensor (N,S[0],S[1],C+5*B) bbox [x,y,w,h]
        """
        image, target = super().__getitem__(index)
        image = to_tensor(image)
        c, h, w = image.shape 
        image = self.transform(image)
        object_list = target['annotation']["object"]

        bbox = [(classes[i['name']],i['bndbox']) for i in object_list]
        bbox = [(i[0],[int(i[1]['xmin']), int(i[1]['ymin']),int(i[1]['xmax']),int(i[1]['ymax'])]) for i in bbox]
        tgt = self._xml2yolo(bbox, h, w)
        # C,Y,X = image.shape
        return image, tgt

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