import copy
import logging
import os

import kornia as K
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import CosineSimilarity
from tqdm import tqdm

# file with read data from DAVIS dataset
import DAVIS_dataset as ld
# ColorAttention class to exctract color information
from modules import EMA, ColorAttention, ImageFeatures, UNet_conditional
from utils import *
from piq import SSIMLoss
from lab_vgg import *
# from ViT import Vit_neck

# Set the random seed
seed = 2023
torch.manual_seed(seed)

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# Difusion example
# https://github.com/dome272/Diffusion-Models-pytorch

# Difussion pre trained
# https://huggingface.co/lambdalabs/sd-image-variations-diffusers

class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=128, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, gray_img, cfg_scale=3, in_ch=3, create_img=True):
        # logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, in_ch, int(self.img_size/16), int(self.img_size/16))).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()

        if create_img:
            y = scale_0_and_1(gray_img)
            try:
                _,u,v = torch.split(x, 1, dim=1)
            except:
                u,v = torch.split(x, 1, dim=1)
            x = torch.cat([y[:,:1], u, v], 1)
            x = tensor_lab_2_rgb(x)

        return x

# def load_vgg_model(out_size=1024, img_size=128, svd_model_name="VGG_20230227_223822"):
#     """
#     Load the pretrained model (VGG in YUV color space), and
#     return the model.
#     """
#     if svd_model_name.split("_")[0] == "VGG":
#         vgg_yuv = VGG_19_YUV(out_size=out_size, img_size=img_size)
#     else:
#         vgg_yuv = Vit_neck()

#     saved_model = os.path.join("models", svd_model_name, f"ckpt.pt")
#     model_wights = torch.load(saved_model)
#     vgg_yuv.load_state_dict(model_wights)

#     return vgg_yuv
    
def train(args):
    setup_logging(args.run_name)
    
    # Dataloader info
    dataLoader = ld.ReadData()
    device = args.device
    dataloader = dataLoader.create_dataLoader(args.dataset_path, args.image_size, args.batch_size, shuffle=True, rgb=False)

    # Models
    model = UNet_conditional(c_in=3, c_out=3, time_dim=args.time_dim).to(device)
    # feature_model = ImageFeatures(inc_ch=3, out_size=args.time_dim).to(device)
    feature_model = load_vgg_model(out_size=args.time_dim, img_size=args.image_size).to(device)
    feature_model.out.eval()
    
    # feature_model = Vit_neck(args.batch_size, args.image_size, args.time_dim).to(device)
    diffusion = Diffusion(img_size=args.image_size, device=device)

    # Optimizers
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    mse = nn.MSELoss()
    # mae = nn.L1Loss()
    # SSIM = SSIMLoss(data_range=1.)
    
    # EMA and log setings
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            img, img_gray, img_color, next_frame = create_samples(data)
            #img = ground truth, img_gray the version monocromatric and img_color the keyframe.

            # Select thw 2 imagens to training
            input_img = img_color.to(device) # Gray Image
            color_img = img.to(device) # Ground truth of the Gray Img        

            # Get the representaion of gray scale image
            labels = feature_model(input_img)

            # key_labels = feature_model(img_color)

            # Generate the noise using the ground truth image
            t = diffusion.sample_timesteps(color_img.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(color_img, t)

            # Predict noise from label (gray representation) + actual noise step
            predicted_noise = model(x_t, t, labels)
            # keyframe_noise = model(x_t, t, key_labels)

            # Loss between noie of prediction and next frame
            # loss_keyframe = mae(keyframe_noise, predicted_noise)

            # Loss betwen noise add from fiddusion and predict by model
            loss_u = mse(predicted_noise[:,1], noise[:,1])

            loss_v = mse(predicted_noise[:,2], noise[:,2])

            loss = loss_u + loss_v

            # Gradient Steps and EMA model criation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 20 == 0:
            # l = len(labels)
            l = 5
            if len(labels) < l:
                l = len(labels)
            
            # Make the loss comparing the prediction of original img and gray img
            sampled_images = diffusion.sample(model, n=l, labels=labels[:l], gray_img=img_gray[:l], in_ch=3)
            ema_sampled_images = diffusion.sample(ema_model, n=l, labels=labels[:l], gray_img=img_gray[:l], in_ch=3)

            plot_images(sampled_images)
            # img_out = tensor_lab_2_rgb(sampled_images).to("cpu")
            # print("Convert Img")
            # plot_images(img_out)
            # print(f"sampled_images max:{sampled_images.max()} / min:{sampled_images.min()}")
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
            torch.save(feature_model.state_dict(), os.path.join("models", args.run_name, f"feature.pt"))


def launch():
    import argparse
    model_name = get_model_time()
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.run_name = f"DDPM_{model_name}"
    args.epochs = 401
    args.batch_size = 6
    args.image_size = 128
    args.time_dim = 1024
    # args.dataset_path = r"C:\video_colorization\data\train\DAVIS_val"
    args.dataset_path = r"C:\video_colorization\data\train\mini_DAVIS"
    # args.dataset_path = r"C:\video_colorization\data\train\DAVIS"
    # args.dataset_path = r"C:\video_colorization\data\train\rallye_DAVIS"
    args.device = "cuda"
    args.lr = 1e-3
    train(args)

if __name__ == '__main__':
    launch()
    print("Done")
