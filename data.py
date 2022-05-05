import torch
from torch import nn
from torchvision.models import vgg16
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import datasets, transforms
import cv2
import matplotlib.pyplot as plt

DATA_ROOT = r"E:\Deep Learning Projects\datasets\kitti_object_detection"
ts = transforms.Compose([transforms.Resize((384, 1248)), transforms.ToTensor()])
ds = datasets.Kitti(DATA_ROOT, True, transform=ts)
img = ds[0][0].unsqueeze(0)
