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
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda", img_size=224, max_ch_deep=128, net_dim=64):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        # self.inc = DoubleConv(c_in, 32)
        self.down1 = Down(3, net_dim)
        # self.sa1 = SelfAttention(net_dim, img_size//2)
        self.down2 = Down(net_dim, net_dim*2)
        self.sa2 = SelfAttention(net_dim*2, img_size//4)
        self.down3 = Down(net_dim*2, net_dim*2)
        self.sa3 = SelfAttention(net_dim*2, img_size//8)
        
        self.bot1 = DoubleConv(net_dim*2, max_ch_deep)
        self.bot2 = DoubleConv(max_ch_deep, net_dim*2)

        self.up1 = Up(net_dim*4, net_dim)
        self.sa4 = SelfAttention(net_dim, img_size//4)
        self.up2 = Up(net_dim*2, net_dim)
        # self.sa5 = SelfAttention(net_dim, img_size//2)
        self.up3 = Up(net_dim+3, net_dim//2)
        self.outc = nn.Sequential(
            nn.Conv2d(net_dim//2, c_out, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):

        # x1 = self.inc(x)
        x1 = x
        x2 = self.down1(x1)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)   

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)

        x = self.up1(x4, x3)
        x = self.sa4(x)
        x = self.up2(x, x2)
        # x = self.sa5(x)
        x = self.up3(x, x1)
        
        output = self.outc(x)
        return output


##### Evaluate the model
#load the dataloader
import old.DAVIS_dataset as ld
dataLoader = ld.ReadData()
import argparse
from tqdm import tqdm

model_name = get_model_time()
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
args.batch_size = 7
args.image_size = 224
args.time_dim = 1024
args.run_name = f"POS_{model_name}"
args.lr = 1e-3
args.epochs = 501
args.net_dim = 220

### Dataset load
root_model_path = r"C:\video_colorization\diffusion\unet_model"
date_str = "UNET_d_20230512_005409"
device = "cuda"
used_dataset = "mini_DAVIS"
dataroot = f"C:/video_colorization/data/train/{used_dataset}"
path_colorized_frames = f"C:/video_colorization/diffusion/temp_result/"

pos_dataroot = os.path.join(path_colorized_frames, used_dataset, date_str)
dataloader = dataLoader.create_dataLoader(dataroot, args.image_size, args.batch_size, shuffle=True, pos_path=pos_dataroot)

### Creating the Folders
pos_path_save = os.path.join("pos_results", args.run_name)
os.makedirs(pos_path_save, exist_ok=True)

pos_path_save_models = os.path.join("pos_model", args.run_name)
os.makedirs(pos_path_save_models, exist_ok=True)

### Model Params to train
pos_process_model = UNet_pos_process(img_size=args.image_size, net_dim=args.net_dim).to(device)

mse = nn.MSELoss()
# SSIM = SSIMLoss(data_range=1.)
# criterion = SSIMLoss(data_range=-1.)
criterion = nn.MSELoss()
optimizer = optim.AdamW(pos_process_model.parameters(), lr=args.lr)

logger = SummaryWriter(os.path.join("runs", args.run_name))
l = len(dataloader)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

### Train Loop
for epoch in range(args.epochs):
    pbar = tqdm(dataloader)
    lp=5
    for i, (data) in enumerate(pbar):
        img, img_gray, img_color, next_frame, pos_color = create_samples(data)

        out = pos_process_model(pos_color)
        loss = criterion(img, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(MSE=loss.item(), epoch=epoch, lr=optimizer.param_groups[0]['lr'])
        logger.add_scalar("SSIM", loss.item(), global_step=epoch * l + i)

    scheduler.step()

    if epoch % 10 == 0:
        img_out = tensor_2_img(out).to("cpu")
        
        plot_images(tensor_2_img(img[:lp]))
        plot_images(img_out[:lp])

        save_images(img_out, os.path.join(pos_path_save, f"{epoch}.jpg"))

        torch.save(pos_process_model.state_dict(), os.path.join(pos_path_save_models, f"ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join(pos_path_save_models, f"optim.pt"))

print("Done")