import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from loss import YoloLoss
from model import YOLOv1
from data import KittiDetection2D
from utils import read_yaml
TRAIN_ROOT = r"E:\Deep Learning Projects\datasets\kitti_object_detection\Kitti\raw\training"
TEST_ROOT = r"E:\Deep Learning Projects\datasets\kitti_object_detection\Kitti\raw\testing"
EPOCHS = 200
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
TRAIN_VAL_SPLIT = 0.8
device = 'cpu' if torch.cuda.is_available() else 'cpu'

img_transforms = transforms.Compose([transforms.Resize((384,1248)), transforms.ToTensor()])

dataset = KittiDetection2D(TRAIN_ROOT, transforms=img_transforms)
train_dataset_len = int(0.8*len(dataset))
val_dataset_len = len(dataset) - train_dataset_len
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_len, val_dataset_len])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

model_cfg = read_yaml('model.yaml')
model = YOLOv1(model_cfg).to(device)
criterion = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

for epoch in range(EPOCHS):
    loop = tqdm(train_dataloader)
    for image, target in loop:
        
        image, target = image.to(device), target.to(device)
        pred = model(image)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()


        loop.set_description(f"Epoch [{epoch}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item())