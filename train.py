from cmath import inf
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from os.path import join as osp
import os

from loss import YoloLoss
from model import get_model
from data import KittiDetection2D
from utils import *
import random

random.seed(10)
torch.manual_seed(10)
print('seed created')
training_config = read_yaml('kitti.yaml')

TRAIN_ROOT = training_config['train_images_path']
TEST_ROOT = training_config['test_image_path']
CKPT_DIR = training_config['ckpt_dir']
EPOCHS = training_config['epochs']
LEARNING_RATE = training_config['learning_rate']
BATCH_SIZE = training_config['batch_size']
NUM_WORKERS = training_config['num_workers']
PIN_MEMORY = training_config['pin_memory']
TRAIN_VAL_SPLIT = training_config['train_val_split']
RESUME = training_config['resume']
CKPT_PATH = training_config['ckpt_path']
RESIZE = tuple(training_config['resize'])
S = tuple(training_config['S'])
B = 2
C = training_config['C']
assert isinstance(S, tuple)

print(f"No. of epochs {EPOCHS}")
print(f"batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Num workers: {NUM_WORKERS}")
print(f"train val split: {TRAIN_VAL_SPLIT}")
print(f"fcl out: {training_config['fcl_out']}")
print(f"dropout: {training_config['dropout']}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(CKPT_DIR, exist_ok=True)

img_transforms = transforms.Compose([transforms.Resize((384,1248)), transforms.ToTensor()])

dataset = KittiDetection2D(TRAIN_ROOT, transforms=img_transforms, S=S, C=C)
train_dataset_len = int(TRAIN_VAL_SPLIT*len(dataset))
val_dataset_len = len(dataset) - train_dataset_len
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_len, val_dataset_len])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=True)

model = get_model(training_config['backbone'])
model = model.to(device)
model_parameters = sum(i.numel() for i in model.parameters())
print(f'Model parameters: {model_parameters}')
criterion = YoloLoss(S=S, C=C)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
epoch = 0
prev_mean_ap = 0
if RESUME:
    state_dict = torch.load(CKPT_PATH)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    epoch = state_dict['epoch']+1
    prev_val_loss = state_dict['loss']
    prev_mean_ap = state_dict['map']
    print(f"Resuming from epoch: {epoch} and loss: {prev_val_loss} and mean_ap: {prev_mean_ap}")


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


for epoch in range(epoch, EPOCHS):
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
    mean_ap = eval(val_dataloader, model)
    print(f'Mean average precision: {mean_ap["map"]}')
    if mean_ap['map'] > prev_mean_ap or epoch%20==0:
        state_dict = {'epoch': epoch,
                      'loss': sum(mean_loss)/len(mean_loss), 
                      'model_state_dict': model.state_dict(), 
                      'optimizer_state_dict': optimizer.state_dict(),
                      'map': mean_ap['map']}
        prev_mean_ap = mean_ap['map']
        save_checkpoint(state_dict, CKPT_DIR)


