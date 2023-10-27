import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def step_ema(self, ema_model, model, step_start_ema=37632):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

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
        x = x.view(-1, self.channels, int(self.size) * int(self.size)).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, int(self.size), int(self.size))


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
    def __init__(self, in_channels, out_channels, emb_dim=1000):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            nn.Dropout2d(p=0.3),
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
    def __init__(self, in_channels, out_channels, emb_dim=1000):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            nn.Dropout2d(p=0.3),
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
    def __init__(self, c_in=3, c_out=384, time_dim=256, device="cuda", max_ch_deep=512, img_size=8, net_dimension=256):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        self.inc = DoubleConv(48+c_out, net_dimension*2)
        self.down1 = Down(net_dimension*2, net_dimension*4)
        self.sa1 = SelfAttention(net_dimension*4, img_size//2)
        self.down2 = Down(192+net_dimension*4, net_dimension*8)
        self.sa2 = SelfAttention(net_dimension*8, img_size//4)
        # self.down3 = Down(768+net_dimension*8, net_dimension*8)
        # self.sa3 = SelfAttention(net_dimension*8, img_size//8)
        
        self.bot1 = DoubleConv(net_dimension*8, max_ch_deep)
        # self.bot1 = DoubleConv(net_dimension*8, 1536)
        # self.bot2 = DoubleConv(1536*2, net_dimension*4)
        self.bot3 = DoubleConv(max_ch_deep, net_dimension*4)

        # self.up1 = Up(net_dimension*8, net_dimension*4)
        # self.sa4 = SelfAttention(net_dimension*4, img_size//2)
        self.up2 = Up(net_dimension*8, net_dimension*2)
        self.sa5 = SelfAttention(net_dimension*2, img_size//2)
        self.up3 = Up(net_dimension*4, net_dimension*2)
        self.sa6 = SelfAttention(net_dimension*2, img_size)
        self.outc = nn.Sequential(
            nn.Conv2d(net_dimension*2, c_out, kernel_size=1)
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
        # if y is not None:
        #     t += torch.flatten(y[:,:25], start_dim=1)

        #Make padding in order to concatenate color with original Img
        # color = y.view(-1, 1536, 5, 5)
        # color = torch.nn.functional.pad(color, (0, 2, 0, 2), "constant", 0)

        x1 = self.inc(torch.cat((x, y[:, :-1].view(-1, 48, 28, 28)),1))
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(torch.cat((x2, y[:, :-1].view(-1, 192, 14, 14)), 1), t)
        x3 = self.sa2(x3)
        # x4 = self.down3(torch.cat((x3, y[:, :-1].view(-1, 768, 7, 7)), 1), t)
        # x4 = self.sa3(x4)   

        x4 = self.bot1(x3)
        # x4 = self.bot2(torch.sum(torch.stack([color, x4]), dim=0))
        # x4 = self.bot2(torch.cat((color, x4), 1))
        x4 = self.bot3(x4)
        # x4 = self.bot3(x4)

        # x = self.up1(x4, x2, t)
        # x = self.sa4(x)
        x = self.up2(x4, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        
        output = self.outc(x)
        return output


if __name__ == '__main__':
    import argparse 
    from ddpm import Diffusion
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.batch_size = 10
    args.image_size = 224
    # net = UNet(device="cpu")
    net = UNet_conditional(c_in=4, c_out=4, time_dim=768, net_dimension=128, img_size=args.image_size//8, device="cuda").to("cuda")
    diffusion = Diffusion(img_size=28, device="cuda", noise_steps=80)

    t = diffusion.sample_timesteps(args.batch_size).to("cuda")
    labels = torch.zeros((args.batch_size, 50, 768)).to("cuda")

    x = torch.ones((args.batch_size,4,28,28)).to("cuda")
    out = net(x, t, labels)

    # t = diffusion.sample_timesteps(x.shape[0]).to(device)
    # x_t, noise = diffusion.noise_images(x, t)

    # out = diff_model(x, t, labels)

    # diffusion = Diffusion(img_size=args.image_size, device="cuda")
    # feature_model = ImageFeatures(out_size=1024).to("cuda")
