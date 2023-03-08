import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import Vit_neck
import torchvision.models as models
from lab_vgg import *

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class ColorAttention(nn.Module):
    def __init__(self, channels, size=64, in_ch=3):
        super(ColorAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.dc = DoubleConv(in_ch, channels)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.ln_out = nn.Sequential(
            nn.LayerNorm([channels*size*size]),
            nn.Linear(channels*size*size, channels*2),
            nn.GELU(),
            nn.Linear(channels*2, channels)
        )

    def forward(self, x):
        # input shape [b, in_ch, size, size]
        x = self.dc(x)
        # x after simple conv. shape [b, 4096, 256]
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # attention_shape [b, 4096, 256] to [b, chanels*size*size]
        attention_value = attention_value.reshape(attention_value.shape[0], -1)
        out = self.ln_out(attention_value)
        # out shape [b, chanels] this change is to concat with t in the diffusion
        return out


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
        # print(f"SELF ATTENT x INPUT: {x.shape}")
        x = x.view(-1, self.channels, int(self.size/16) * int(self.size/16)).swapaxes(1, 2)
        # print(f"SELF ATTENT x shape: {x.shape}")
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # print(f"ATTENTION_VALUE shape: {attention_value.shape}")
        return attention_value.swapaxes(2, 1).view(-1, self.channels, int(self.size/16), int(self.size/16))


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
    def __init__(self, in_channels, out_channels, emb_dim=1024):
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

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=1024):
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

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
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
        self.outc = nn.Sequential(
            nn.Conv2d(128, c_out, kernel_size=1)
        )

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            t += y

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)   

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        
        output = self.outc(x)
        return output


class ImageFeatures(nn.Module):
    def __init__(self, inc_ch=1, out_size=256, img_size=128, svd_model_name="VGG_20230306_151307") -> None:
        super().__init__()

        # Model trained name
        self.svd_model_name = svd_model_name

        # Img size to define the dimension of last vector output
        self.img_size = img_size

        # Load the YUV pre trained model
        self.out = self.__load_vgg_model__()

    def __load_vgg_model__(self):
        """
        Load the pretrained model (VGG in YUV color space), and
        return the model.
        """
        saved_model = os.path.join("models", self.svd_model_name, f"ckpt.pt")
        model_wights = torch.load(saved_model)
        vgg_yuv = VGG_19_YUV(out_size=out_size, img_size=self.img_size).to(device)
        vgg_yuv.load_state_dict(model_wights)
        return vgg_yuv

    def forward(self, x):
        out = self.out(x)
        return out

        

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.batch_size = 4
    args.image_size = 128
    # net = UNet(device="cpu")
    net = UNet_conditional(c_in=2, c_out=2, time_dim=1024, device="cuda")
    # diffusion = Diffusion(img_size=args.image_size, device="cuda")
    feature_model = ImageFeatures(out_size=1024).to("cuda")

    # ViT_model = Vit_neck(image_size=args.image_size, batch_size=args.batch_size).to("cuda")
    # print(sum([p.numel() for p in net.parameters()]))

    x = torch.randn(args.batch_size, 3,  args.image_size,  args.image_size).to("cuda")
    x_color = torch.randn(args.batch_size, 3,  args.image_size,  args.image_size).to("cuda")
    
    labels = feature_model(x_color)

    t = x.new_tensor([500] * x.shape[0]).long().to("cuda")
    # y = x.new_tensor([1] * x.shape[0]).long()
    # print(net(x, t, y).shape)
    
    # out_color = ViT_model(x_color)
    net = net.to("cuda")

    color_model = ImageFeatures(out_size=512).to("cuda")
    print(color_model(x).shape)
