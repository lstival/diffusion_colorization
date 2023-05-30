"""
Evaluation of the model, and save the video colorized
from the example image passed
"""
import os
import torch
import cv2

from tqdm import tqdm

# My patchs
import read_data as ld
from modules import *
from ddpm import *
from utils import *
import VAE as vae
from ViT import Vit_neck

import shutil
# ================ Initial Infos =====================
import argparse
model_name = get_model_time()
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

args.time_dim = 1000
args.noise_steps = 100
args.rgb = True

args.batch_size = 50
args.image_size = 224
args.in_ch=256
args.out_ch = 256
args.net_dimension=128

# dataset = "mini_kinetics"
dataset = "mini_DAVIS"
data_mode = "train"

batch_size = args.batch_size
device = "cuda"
date_str = "UNET_d_20230526_150739"
best_model = False

# List all classes to be evaluated
images_paths = f"C:/video_colorization/data/{data_mode}/{dataset}"
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
    create_gray_videos(dataset, path_gray_video, data_mode)
    list_gray_videos = os.listdir(path_gray_video)


# ================ Read Model =====================
root_model_path = r"C:\video_colorization\diffusion\unet_model"
# date_str = "UNET_20230404_120711"


### Encoder
# feature_model = Encoder(c_in=3, c_out=args.out_ch//2, return_subresults=True, img_size=args.image_size).to(device)
# feature_model = load_trained_weights(feature_model, date_str, "feature")

# ### Color Neck
# color_neck = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=int((args.in_ch//2)*8*8))
# color_neck = load_trained_weights(color_neck, date_str, "vit_neck")
# color_neck.eval()

### Decoder
# decoder = Decoder(c_in=args.in_ch, c_out=3, img_size=args.image_size).to(device)
# decoder = Decoder(c_in=args.out_ch, c_out=3, img_size=args.image_size).to(device)
# decoder = load_trained_weights(decoder, date_str, "decoder")

### Diffusion process
diffusion = Diffusion(img_size=args.image_size//8, device=device, noise_steps=args.noise_steps)
diffusion_model = UNet_conditional(c_in=4, c_out=4, time_dim=args.time_dim, img_size=args.image_size//8,net_dimension=args.net_dimension).to(device)
if best_model:
    diffusion_model = load_trained_weights(diffusion_model, date_str, "best_model")
else:
    diffusion_model = load_trained_weights(diffusion_model, date_str, "ema_ckpt")

diffusion_model.eval()

# ### Labels generation
prompt = Vit_neck().to("cuda")
prompt.eval()

# ================ Loop all videos inside gray folder =====================
pbar = tqdm(list_gray_videos)
for video_name in pbar:
    #Count for frames in video
    count_frame_idx=0
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
    dataloader = dataLoader.create_dataLoader(path_temp_gray_frames, args.image_size, batch_size, pin_memory=False)

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

        # pbar = tqdm(dataloader)
        for i, (data) in enumerate(dataloader):
            # Set the imagens from dataloader
            img, img_gray, img_color, _ = create_samples(data)            
            #### Images

            ## Gray Image
            input_img = img_gray.to(device)
            l = input_img.shape[0]

            ## Labels to create sample from noise
            labels = prompt(input_img)

            ### Diffusion (due the noise version of input and predict)
            x = diffusion.sample(diffusion_model, labels=labels, n=l, in_ch=4, create_img=False).half()

            ### Decoder the output of diffusion
            sampled_images = vae.latents_to_pil(x)

            for img_idx in range(len(sampled_images)):
                if args.rgb:
                    save_images((sampled_images[img_idx]), os.path.join(colored_frames_save,  f"{str(count_frame_idx).zfill(5)}.jpg"))
                else:
                    save_images(tensor_lab_2_rgb(sampled_images[img_idx].unsqueeze(0)), os.path.join(colored_frames_save,  f"{str(count_frame_idx).zfill(5)}.jpg"))
                count_frame_idx+=1
            
            torch.cuda.empty_cache()

    frame_2_video(colored_frames_save, f"{colored_video_path}/{video_name}_colored.mp4", img_start_name=None)

print("Evaluation Finish")