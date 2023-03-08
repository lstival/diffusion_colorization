#load the modules (networks)
from modules import *

#loade the diffusion model
from ddpm import *

#load the dataloader
import DAVIS_dataset_pos as ld

#PSNR metric
import torchmetrics

# Utils Methods
from utils import *

import pandas as pd

image_size = 128
batch_size = 1
device = "cuda"

##### Read Data
dataLoader = ld.ReadData()
used_dataset = "rallye_DAVIS"

dataroot = f"C:/video_colorization/data/train/{used_dataset}"

dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False, rgb=False)

##### Evaluate the model

root_model_path = r"C:\video_colorization\diffusion\models"
date_str = "DDPM_20230218_090502"

model_path = os.path.join(root_model_path, date_str, "ckpt.pt")
feature_path = os.path.join(root_model_path, date_str, "feature.pt")

model_weights = torch.load(model_path)
feature_weights = torch.load(feature_path)

# https://github.com/dome272/Diffusion-Models-pytorch/blob/main/utils.py

import argparse
model_name = get_model_time()
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
args.batch_size = 1
args.image_size = 128
args.time_dim = 512

# Reading models
# ViT_model = Vit_neck(image_size=args.image_size, batch_size=8).to(device)
feature_model = ImageFeatures(out_size=args.time_dim).to(device)
model = UNet_conditional(time_dim=args.time_dim, c_in=2, c_out=2).to(device)
diffusion = Diffusion(img_size=args.image_size, device=device)

# Definir o modelo em modo de avaliação
model.load_state_dict(model_weights)
feature_model.load_state_dict(feature_weights)

# Set the models in eval mode
model.eval()
feature_model.eval()

# Dict with a list of each metric comparing original with ground truth
metrics = {}
metrics["PSNR"] = []

# metrics = eval_dataset(metrics)
PSNR = torchmetrics.PeakSignalNoiseRatio()

save_path = f"evals/{date_str}/{used_dataset}"
os.makedirs(save_path, exist_ok=True)

img_count = 0
with torch.no_grad():
    pbar = tqdm(dataloader)
    for i, (data) in enumerate(pbar):
        # Set the imagens from dataloader
        img, img_gray, img_color, next_frame = create_samples(data)

        # Use Vit to create label to produce sample from noise
        labels = feature_model(img_gray)

        # Reconstruct frame from label produced by VGG
        sampled_images = diffusion.sample(model, n=args.batch_size, labels=labels, gray_img=img_gray[:,:1], in_ch=2)

        for img_idx in range(args.batch_size):
    
            gray_img2 = ((tensor_2_img(img_gray[img_idx]))).type(torch.uint8)
            save_images(gray_img2, os.path.join(save_path, f"gray_{str(i).zfill(5)}.jpg"))
            save_images(sampled_images[img_idx], os.path.join(save_path, f"out_{str(i).zfill(5)}.jpg"))

            # Compare the images
            psnr = PSNR(img.squeeze().to("cpu"), sampled_images.squeeze().to("cpu"))
            metrics["PSNR"].append(psnr.item())

# Saving the metrics
save_path_metrics = f"metrics/{date_str}"
os.makedirs(save_path_metrics, exist_ok=True)

df_metrics = pd.DataFrame.from_dict(metrics)
df_metrics.to_csv(os.path.join(save_path_metrics, "metrics.csv"))


# Loop para todos os videos em GRAY do dataset
# Salva os frames gray
# Le esses frames 1 por 1 e colore eles
# Salva os frames coloridos
# Le os frames coloridos e gera o vídeo