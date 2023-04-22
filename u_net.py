import torch
import os
from utils import *
from ddpm import *
import read_data as ld

# Load the Network to increase the colorization
import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import Vit_neck
import torchvision.models as models

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        """
        channels: How much chanels the input have
        size: dimension of the matrix (HxW) of NxCxHxW
        """
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
            DoubleConv(in_channels, out_channels, in_channels),
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
    
class Encoder(nn.Module):
    def __init__(self, c_in=3, c_out=128, img_size=128, return_subresults=False) -> None:
        super().__init__()
        self.subresults = return_subresults
        self.inc = DoubleConv(c_in, c_out//4)
        self.down1 = Down(c_out//4, c_out//2)
        self.sa1 = SelfAttention(c_out//2, img_size//2)
        self.down2 = Down(c_out//2, c_out)
        self.sa2 = SelfAttention(c_out, img_size//4)
        self.down3 = Down(c_out, c_out)
        self.sa3 = SelfAttention(c_out, img_size//8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)   

        if self.subresults:
            return x4, (x3, x2, x1)
        else:
            return x4

class Decoder(nn.Module):
    def __init__(self, c_in=128, c_out=3, img_size=128, vit_neck=False) -> None:
        super().__init__()
        self.vit_neck = vit_neck
        self.inc = DoubleConv(c_in, c_in//2)
        self.up1 = Up(c_in, c_in//4)
        self.sa4 = SelfAttention(c_in//4, img_size//4)
        self.up2 = Up(c_in//2, c_in//8)
        self.sa5 = SelfAttention(c_in//8, img_size//2)
        self.up3 = Up(c_in//4, c_in//4)
        self.sa6 = SelfAttention(c_in//4, img_size)
        self.outc = nn.Sequential(
            nn.Conv2d(c_in//4, c_out, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Set the skip connections tensors
        x, skips = x
        # x4 = x.clone()
        x3,x2,x1 = skips

        # Forward the process
        if self.vit_neck:
            x = self.inc(x)
        x = self.up1(x, x3)
        x = self.sa4(x)
        x = self.up2(x, x2)
        # x = self.sa5(x)
        x = self.up3(x, x1)
        # x = self.sa6(x)

        output = self.outc(x)
        return output

def load_vgg_model(batch_size, out_size=1024, img_size=128, svd_model_name="VIT_20230306_161039"):
    """
    Load the pretrained model (VGG in YUV color space), and
    return the model.
    """
    if svd_model_name.split("_")[0] == "VGG":
        vgg_yuv = VGG_19_YUV(out_size=out_size, img_size=img_size)
    else:
        vgg_yuv = Vit_neck(batch_size=batch_size, image_size=img_size, out_chanels=out_size)

    saved_model = os.path.join("models", svd_model_name, f"ckpt.pt")
    model_wights = torch.load(saved_model)
    vgg_yuv.load_state_dict(model_wights)

    return vgg_yuv

if __name__ == "__main__":
    print("main")

    device = "cuda"
    in_ch = 128
    image_size=128
    batch_size=16

    feature_model = Encoder(c_in=3, c_out=in_ch//2, return_subresults=True, img_size=image_size).to(device)

    decoder = Decoder(c_in=in_ch, c_out=3, img_size=image_size).to(device)

    input_img = torch.zeros((batch_size,3,image_size,image_size)).to(device)

    gt_out, skips = feature_model(input_img)

    dec_out = decoder((gt_out, skips))

    print(f"dec_out shape: {dec_out.shape}")
