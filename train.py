from cmath import inf
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from os.path import join as osp
import os

from loss import YoloLoss
from model import YOLOv1
from data import KittiDetection2D
from utils import *

TRAIN_ROOT = r"E:\Deep Learning Projects\datasets\kitti_2d_object_detection\Kitti\raw\training"
TEST_ROOT = r"E:\Deep Learning Projects\datasets\kitti_2d_object_detection\Kitti\raw\testing"
CKPT_DIR = r"ckpt_dir"
os.makedirs(CKPT_DIR, exist_ok=True)
EPOCHS = 200
LEARNING_RATE = 1e-6
BATCH_SIZE = 8
NUM_WORKERS = 0
PIN_MEMORY = False
TRAIN_VAL_SPLIT = 0.00201
RESUME = False
CKPT_PATH = ""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

S = (6,20)
B = 2
C = 9

img_transforms = transforms.Compose([transforms.Resize((384,1248)), transforms.ToTensor()])

dataset = KittiDetection2D(TRAIN_ROOT, transforms=img_transforms)
train_dataset_len = int(TRAIN_VAL_SPLIT*len(dataset))
val_dataset_len = len(dataset) - train_dataset_len
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_len, val_dataset_len])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)

model_cfg = read_yaml('model.yaml')
model = YOLOv1(model_cfg).to(device)
criterion = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
if RESUME:
    state_dict = torch.load(CKPT_PATH)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    epoch = state_dict['epoch']
    prev_val_loss = state_dict['loss']
    print(f"Resuming from epoch: {epoch} and loss: {prev_val_loss}")


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
prev_val_loss = inf
for epoch in range(EPOCHS):
    loop = tqdm(train_dataloader)
    mean_loss = []
    for image, target in loop:
        
        image, target = image.to(device), target.to(device)
        pred = model(image)
        loss = criterion(pred, target)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())
    
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

    print("starting validation ...")
    # loop = tqdm(val_dataloader)
    # mean_loss = []
    # model.eval()
    # for image, target in loop:
    #     with torch.no_grad():
    #       image, target = image.to(device), target.to(device)
    #       pred = model(image)
    #       loss = criterion(pred, target)
    #       mean_loss.append(loss.item())

    # print(f"Mean validation loss was {sum(mean_loss)/len(mean_loss)}")
    # if prev_val_loss > sum(mean_loss)/len(mean_loss):
    #     state_dict = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'loss':sum(mean_loss)/len(mean_loss)}
    #     torch.save(state_dict, osp(CKPT_DIR, f"yolo_v1_ckpt_{epoch}.pth"))
    #     prev_val_loss = sum(mean_loss)/len(mean_loss)

    eval(train_dataloader, model)


