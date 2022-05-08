import torch
from torch.utils.data import Dataset
import os
from os.path import join as osp
from PIL import Image
from torchvision.transforms import transforms

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

    def __init__(self, root, S=(11,24), B=2, C=9, transforms=None):
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
        x = int((x1+x2)/2)/self.w
        y = int((y1+y2)/2)/self.h
        h = int((y2-y1))/self.h
        w = int((x2-x1))/self.w
        cell = int(y*self.S[0]), int(x*self.S[1])
        x = x*self.S[1]-cell[1]
        y = y*self.S[0]-cell[0]
        h = self.S[0]*h
        w = self.S[1]*w
        return cell,x,y,w,h
