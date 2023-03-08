import torch
import os
from utils import *
from ddpm import *

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
    
class Up2(nn.Module):
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

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class UNet(nn.Module):
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

class Encoder(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=1024, device="cuda", max_ch_deep=512) -> None:
        super().__init__()
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 64)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 32)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 16)

    def forward(self, x, x_skips):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.sa1(x2)
        x3 = self.down2(x2)
        x3 = self.sa2(x3)
        x4 = self.down3(x3)
        x4 = self.sa3(x4)   

        return x4

class Decoder(nn.Module):
    def __init__(self, c_in=2, c_out=2, time_dim=1024, device="cuda", max_ch_deep=512) -> None:
        super().__init__()
        self.up1 = Up2(c_in, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up2(128, 128)
        self.sa5 = SelfAttention(128, 32)
        self.up3 = Up2(128, 128)
        self.sa6 = SelfAttention(128, 64)
        self.up4 = Up2(128, 128)
        self.outc = nn.Sequential(
            nn.Conv2d(128, c_out, kernel_size=1)
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.sa4(x)
        x = self.up2(x)
        x = self.sa5(x)
        x = self.up3(x)
        x = self.sa6(x)
        x = self.up4(x)
        
        output = self.outc(x)
        return output

def load_vgg_model(out_size=1024, img_size=128, svd_model_name="VIT_20230306_161039", batch_size=batch_size):
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

def train():
    ### Load dataset 
    import DAVIS_dataset as ld
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=True)

    ### Models
    # encoder = Encoder().to(device)
    decoder = Decoder(c_in=512).to(device)
    decoder.train()

    prompt = load_vgg_model(out_size=time_dim, img_size=image_size, svd_model_name="VIT_20230309_235259", batch_size=batch_size).to(device) # Vit to features from color sample
    feature_model = load_vgg_model(out_size=time_dim, img_size=image_size, svd_model_name="VGG_20230306_151307", batch_size=batch_size).to(device) # VGG as encoder
    feature_model.train()
    
    diffusion = Diffusion(img_size=image_size, device=device, noise_steps=noise_steps)
    diffusion_model = UNet_conditional(c_in=512, c_out=512, time_dim=time_dim).to(device)
    diffusion_model.train()

    ### Optimizers
    mse = nn.MSELoss()

    params_list = list(diffusion_model.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params_list, lr=lr)
    # optimizer_deco = optim.AdamW(decoder.parameters(), lr=lr)

    ### Train Loop
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            img, img_gray, img_color, next_frame = create_samples(data)

            # Avoid break during the sample of denoising
            l = img.shape[0]
            
            # Select thw 2 imagens to training
            input_img = img_color.to(device) # Gray Image
            gt_img = img.to(device) # Ground truth of the Gray Img   

            ### Labels to create sample from noise
            labels = prompt(input_img)

            ### Encoder (create feature representation)
            gt_out = feature_model(gt_img)

            ### Generate the noise using the ground truth image
            t = diffusion.sample_timesteps(gt_out.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(gt_out, t)

            ### Predict the noise 
            predicted_noise = diffusion_model(x_t, t, labels)

            ### Meansure the difference between noise predicted and realnoise
            loss = mse(predicted_noise, noise)

            ### Diffusion (due the noise version of input and predict)
            dif_out = diffusion.sample(diffusion_model, n=l, labels=labels, gray_img=img_gray, in_ch=512, create_img=False)

            ### Decoder (create the expected sample using denoised feature space)
            dec_out = decoder(dif_out)
            y = scale_0_and_1(img_gray[:,:1])
            x = torch.cat([y, dec_out],1)

            loss += mse(x[:,1:], gt_img[:,1:])

            # Gradient Steps and EMA model criation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # pbar.set_postfix(MSE=loss.item())

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            l = 5
            if len(labels) < l:
                l = len(labels)

            plot_img = tensor_lab_2_rgb(x[:l])
            ### Creating the Folders
            pos_path_save = os.path.join("unet_results", run_name)
            os.makedirs(pos_path_save, exist_ok=True)

            pos_path_save_models = os.path.join("unet_model", run_name)
            os.makedirs(pos_path_save_models, exist_ok=True)

            ### Ploting and saving
            # sampled_images = diffusion.sample(model, n=l, labels=labels[:l], gray_img=img_gray[:l], in_ch=2)
            plot_images(tensor_lab_2_rgb(gt_img[:l]))
            plot_images(plot_img)
            save_images(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))
            torch.save(diffusion_model.state_dict(), os.path.join("unet_model", run_name, f"ckpt.pt"))
            torch.save(feature_model.state_dict(), os.path.join("unet_model", run_name, f"feature.pt"))


if __name__ == "__main__":
    model_name = get_model_time()
    run_name = f"UNET_{model_name}"
    epochs = 201
    noise_steps = 200

    image_size=128
    batch_size=8
    device="cuda"
    lr=2e-3
    time_dim=1024
    dataroot = r"C:\video_colorization\data\train\mini_DAVIS"
    train()
    print("Done")

    # decoder = Decoder(c_in=512)
    # a = torch.ones([2, 512, 8, 8])
    # decoder(a)

    # x4.shape
    # torch.Size([4, 256, 16, 16])

    # img = torch.ones((4,3,128,128))
    # out = model(img)