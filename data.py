import torch
from torch.utils.data import Dataset
import os
from os.path import join as osp
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image

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


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            # image = self.transform(image)
            image = self.transform(image)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
