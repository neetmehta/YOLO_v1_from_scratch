import torch
from torch.utils.data import Dataset
import os
from os.path import join as osp
from PIL import Image
from torchvision.transforms import transforms
import cv2

img_transforms = transforms.Compose([ transforms.ToTensor()])

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

    def __init__(self, root, transforms=None):
        super(KittiDetection2D, self).__init__()
        self.image_dir = osp(root, "image_2")
        self.label_dir = osp(root, "label_2")
        self.image_list = os.listdir(osp(root, "image_2"))
        self.label_list = os.listdir(osp(root, "label_2"))
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = Image.open(osp(self.image_dir, self.image_list[index]))
        label_pth = osp(self.label_dir, self.label_list[index])
        box_list = []
        with open(label_pth, 'r') as f:
            for i in f.readlines():
                box_list.append(self._parse_label(i))
            
        if self.transforms:
            image = self.transforms(image)

        return image, box_list

    def _parse_label(self, label):
        label = label.split()
        obj_class = label_map[label[0]]
        x1, y1, x2, y2 = int(float(label[4])), int(float(label[5])), int(float(label[6])), int(float(label[7]))
        return obj_class, x1, y1, x2, y2

ds = KittiDetection2D(r"E:\Deep Learning Projects\datasets\kitti_object_detection\Kitti\raw\training", img_transforms)
label = ds[2][1][0]
print(label[0])
img = cv2.imread(r"E:\Deep Learning Projects\datasets\kitti_object_detection\Kitti\raw\training\image_2\000002.png")
cv2.rectangle(img, (label[1], label[2]), (label[3], label[4]), color=(254,0,0), thickness=1)
cv2.imshow("img", img)
cv2.waitKey(0)