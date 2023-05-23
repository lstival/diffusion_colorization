#PSNR metric
import torchmetrics
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
# from frechetdist import frdist

# Utils Methods
from utils import *
from tqdm import tqdm

# Data Manipulation
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
import pandas as pd

def Average(lst):
    return sum(lst) / len(lst)

### Parameters
image_size = 224
time_dim = 1000
device = "cuda"

##### Read Data
used_dataset = "DAVIS_val"
date_str = "UNET_20230406_182152"

### Ground Truth Frames
gt_dataroot = f"C:/video_colorization/data/train/{used_dataset}"
# gt_dataloader = dataLoader.create_dataLoader(gt_dataroot, batch_size, shuffle=False)

### Colorized Frames
colorized_images_path = r"C:\video_colorization\diffusion\temp_result"
colorized_dataroot = os.path.join(colorized_images_path, used_dataset, date_str)
# colorized_dataloader = dataLoader.create_dataLoader(colorized_dataroot, batch_size, shuffle=False)

# https://github.com/dome272/Diffusion-Models-pytorch/blob/main/utils.py

### Name of videos present in the dataset
filenames = os.listdir(colorized_dataroot)

# Dict with a list of each metric comparing original with ground truth
metrics = {}
# metrics["PSNR"] = []

# metrics = eval_dataset(metrics)
PSNR = torchmetrics.PeakSignalNoiseRatio()
SSIM = StructuralSimilarityIndexMeasure(data_range=1.0)
fid = FrechetInceptionDistance(feature=64)
# frdist

save_path = f"evals/{date_str}/{used_dataset}"
os.makedirs(save_path, exist_ok=True)

pbar = tqdm(filenames)
for filename in pbar:
    pbar.set_description(f"Processing {filename}")

    metrics[filename.split(".")[0]] = {"PSNR": [], "SSIM": [], "FID": []}

    total_frames = len(os.listdir(os.path.join(gt_dataroot, filename.split(".")[0])))

    fid_gt = []
    fid_clr = []

    # Loop for each frame in the video
    for idx_frame in range(total_frames):

        # Load the images and resize them
        gt_img = Image.open(os.path.join(gt_dataroot, filename.split(".")[0], str(idx_frame).zfill(5)+".jpg")).resize((image_size, image_size))
        clr_img = Image.open(os.path.join(colorized_dataroot, filename, str(idx_frame).zfill(5)+".jpg")).resize((image_size, image_size))

        gt_tensor = pil_to_tensor(gt_img)
        clr_tensor = pil_to_tensor(clr_img)

        # Salve img to calculate fid
        fid_gt.append(gt_tensor)
        fid_clr.append(clr_tensor)

        ## Compare the images

        # calculate PSNR
        psnr = PSNR(gt_tensor, clr_tensor)
        # calculate SSIM
        ssim = SSIM(scale_0_and_1(gt_tensor.unsqueeze(0)), scale_0_and_1(clr_tensor.unsqueeze(0)))

        ## Save values in temp lists
        metrics[filename.split(".")[0]]["PSNR"].append(psnr.item())
        metrics[filename.split(".")[0]]["SSIM"].append(ssim.item())

    # Calculate FID
    fid.update(torch.stack(fid_gt, dim=0), real=True)
    fid.update(torch.stack(fid_clr, dim=0), real=False)
    # fid = frdist(gt_tensor, clr_tensor)
    fid_out = fid.compute().item()
    metrics[filename.split(".")[0]]["FID"].append(fid_out)

# Saving the metrics
save_path_metrics = f"metrics/{date_str}"
os.makedirs(save_path_metrics, exist_ok=True)

temp_df_metrics = pd.DataFrame.from_dict(metrics)
df_metrics = pd.DataFrame(columns=["PSNR", "SSIM", "FID"])

df_metrics["PSNR"] = temp_df_metrics.apply(lambda x: Average(x[0]), axis=0)
df_metrics["SSIM"] = temp_df_metrics.apply(lambda x: Average(x[1]), axis=0)
df_metrics["FID"] = temp_df_metrics.apply(lambda x: Average(x[2]), axis=0)
df_metrics.to_csv(os.path.join(save_path_metrics, "metrics.csv"))

