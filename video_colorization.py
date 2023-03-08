"""
Evaluation of the model, and save the video colorized
from the example image passed
"""
import os
import torch
import cv2

from tqdm import tqdm

# My patchs
import DAVIS_dataset_pos as ld
from modules import *
from ddpm import *
from utils import *

import shutil
# ================ Initial Infos =====================
import argparse
model_name = get_model_time()
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()
args.batch_size = 1
args.image_size = 64
args.time_dim = 1024

dataset = "mini_DAVIS"
batch_size = args.batch_size
device = "cuda"

# List all classes to be evaluated
images_paths = f"C:/video_colorization/data/train/{dataset}"
img_classes = os.listdir(images_paths)

# ================ Read Video =====================

# Temp folder to save the video frames (delete after process)
temp_path = f"temp/{dataset}/"
os.makedirs(temp_path, exist_ok=True)

# Path where is the gray version of videos
path_gray_video = f"C:/video_colorization/data/videos/{dataset}_gray/"
try:
    list_gray_videos = os.listdir(path_gray_video)
except FileNotFoundError:
    # Create the gray version of the videos
    create_gray_videos(dataset, path_gray_video)
    list_gray_videos = os.listdir(path_gray_video)


# ================ Read Model =====================
root_model_path = r"C:\video_colorization\diffusion\models"
date_str = "DDPM_20230224_020521"

# Path to load the weights
model_path = os.path.join(root_model_path, date_str, "ckpt.pt")
feature_path = os.path.join(root_model_path, date_str, "feature.pt")

# Load the weights
model_weights = torch.load(model_path)
feature_weights = torch.load(feature_path)

# Set features weights
feature_model = ImageFeatures(out_size=args.time_dim).to(device)
feature_model.load_state_dict(feature_weights)
feature_model.eval()

# Set revser diffusion model weights
model = UNet_conditional(time_dim=args.time_dim, c_in=2, c_out=2).to(device)
model.load_state_dict(model_weights)

# Create the diffusion process
diffusion = Diffusion(img_size=args.image_size, device=device)

# ================ Loop all videos inside gray folder =====================
pbar = tqdm(list_gray_videos)
for video_name in pbar:
    pbar.set_description(f"Processing: {video_name}")

    vidcap = cv2.VideoCapture(f"{path_gray_video}{video_name}")
    success,image = vidcap.read()
    count = 0
    list_frames = []

    path_temp_gray_frames = f"{temp_path}{video_name.split('.')[0]}"
    if not os.path.exists(path_temp_gray_frames):

        os.makedirs(f"{path_temp_gray_frames}/images/", exist_ok=True)

        while success:
            cv2.imwrite(f"{path_temp_gray_frames}/images/{str(count).zfill(5)}.jpg", image)     # save frame as JPEG file      
            success,image = vidcap.read()
            list_frames.append(image)
            count += 1

    # ================ Read images to make the video =====================
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(path_temp_gray_frames, args.image_size, batch_size)

    # ============== Frame Production ===================
    imgs_2 = []
    imgs_2_gray = []

    outs = []
    # path to save colored frames
    colored_frames_save = f"temp_result/{dataset}/{date_str}/{video_name}/"

    if os.path.exists(colored_frames_save):
        shutil.rmtree(colored_frames_save)

    os.makedirs(colored_frames_save, exist_ok=True)

    # path so save videos
    colored_video_path = f"videos_output/{date_str}/{video_name}/"
    os.makedirs(colored_video_path, exist_ok=True)

    img_count = 0
    with torch.no_grad():
        model.eval()
        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            # Set the imagens from dataloader
            img, img_gray, img_color, next_frame = create_samples(data)

            # Use Vit to create label to produce sample from noise
            labels = feature_model(img_gray)

            # Reconstruct frame from label produced by VGG
            sampled_images = diffusion.sample(model, n=args.batch_size, labels=labels, gray_img=img_gray[:,:1], in_ch=2)

            for img_idx in range(args.batch_size):
                save_images(sampled_images[img_idx], os.path.join(colored_frames_save,  f"{str(i).zfill(5)}.jpg"))

    frame_2_video(colored_frames_save, f"{colored_video_path}/{video_name}_colored.mp4", img_start_name=None)

print("Evaluation Finish")