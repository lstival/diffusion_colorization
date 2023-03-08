import torch
import os
from utils import *
from ddpm import *

# Load the images
## Todo Dataloader that read image colorized and grond truth (to compare new colorization and original)
# Load the ddpm model

# Load the Network to increase the colorization
import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import Vit_neck
import torchvision.models as models

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x

class UNet_pos_process(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda", max_ch_deep=512):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 64)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 32)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 16)
        
        self.bot1 = DoubleConv(256, max_ch_deep)
        self.bot2 = DoubleConv(max_ch_deep, max_ch_deep)
        self.bot3 = DoubleConv(max_ch_deep, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 32)
        self.up2 = Up(256, 128)
        self.sa5 = SelfAttention(128, 64)
        self.up3 = Up(192, 128)
        self.sa6 = SelfAttention(64, 32)
        self.outc = nn.Sequential(
            nn.Conv2d(128, c_out, kernel_size=1)
        )

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)   

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3)
        x = self.sa4(x)
        x = self.up2(x, x2)
        x = self.sa5(x)
        x = self.up3(x, x1)
        
        output = self.outc(x)
        return output


##### Evaluate the model
#load the dataloader
import DAVIS_dataset_pos as ld
dataLoader = ld.ReadData()
import argparse
from tqdm import tqdm

model_name = get_model_time()
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
args.batch_size = 8
args.image_size = 128
args.time_dim = 512
args.run_name = f"POS_{model_name}"
args.lr = 3e-3
args.epochs = 101

### Dataset load
root_model_path = r"C:\video_colorization\diffusion\models"
date_str = "DDPM_20230218_090502"
device = "cuda"
used_dataset = "rallye_DAVIS"
dataroot = f"C:/video_colorization/data/train/{used_dataset}"

pos_dataroot = os.path.join("C:/video_colorization/diffusion/evals", date_str, used_dataset)
dataloader = dataLoader.create_dataLoader(dataroot, args.image_size, args.batch_size, shuffle=False, pos_path=pos_dataroot)

### Creating the Folders
pos_path_save = os.path.join("pos_results", args.run_name)
os.makedirs(pos_path_save, exist_ok=True)

pos_path_save_models = os.path.join("pos_model", args.run_name)
os.makedirs(pos_path_save_models, exist_ok=True)

### Model Params to train
pos_process_model = UNet_pos_process().to(device)

mse = nn.MSELoss()
SSIM = SSIMLoss(data_range=1.)
optimizer = optim.AdamW(pos_process_model.parameters(), lr=args.lr)

logger = SummaryWriter(os.path.join("runs", args.run_name))
l = len(dataloader)

### Train Loop
for epoch in range(args.epochs):
    pbar = tqdm(dataloader)
    for i, (data) in enumerate(pbar):
        img, img_gray, img_color, next_frame, pos_color = create_samples(data)

        out = pos_process_model(pos_color)
        loss = mse(img, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(MSE=loss.item())
        logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

    if epoch % 10 == 0:
        img_out = tensor_lab_2_rgb(out).to("cpu")
        
        plot_images(img_out)
        save_images(img_out, os.path.join(pos_path_save, f"{epoch}.jpg"))

        torch.save(pos_process_model.state_dict(), os.path.join(pos_path_save_models, f"ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join(pos_path_save_models, f"optim.pt"))