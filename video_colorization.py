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
from u_net import *

import shutil
# ================ Initial Infos =====================
import argparse
model_name = get_model_time()
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

args.time_dim = 1024
args.noise_steps = 1000
args.rgb = True

args.batch_size = 8
args.image_size = 64
args.in_ch=256

# dataset = "mini_kinetics"
dataset = "rallye_DAVIS"
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
root_model_path = r"C:\video_colorization\diffusion\unet_model"
# date_str = "UNET_20230404_120711"
date_str = "UNET_k_20230423_140134"

### Encoder
feature_model = Encoder(c_in=3, c_out=args.in_ch//2, return_subresults=True, img_size=args.image_size).to(device)
feature_model = load_trained_weights(feature_model, date_str, "feature")

# ### Color Neck
# color_neck = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=int((args.in_ch//2)*8*8))
# color_neck = load_trained_weights(color_neck, date_str, "vit_neck")
# color_neck.eval()

### Decoder
# decoder = Decoder(c_in=args.in_ch, c_out=3, img_size=args.image_size).to(device)
decoder = Decoder(c_in=args.in_ch, c_out=3, img_size=args.image_size).to(device)
decoder = load_trained_weights(decoder, date_str, "decoder")

### Diffusion process
# diffusion = Diffusion(img_size=8, device=device, noise_steps=args.noise_steps)

# diffusion_model = UNet_conditional(c_in=args.in_ch, c_out=args.in_ch, time_dim=args.time_dim, img_size=8).to(device)
# diffusion_model = load_trained_weights(diffusion_model, date_str, "ckpt")

# ### Labels generation
# prompt = Vit_neck(batch_size=batch_size, image_size=args.image_size, out_chanels=args.time_dim)
# prompt = load_trained_weights(prompt, date_str, "prompt")

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
        # diffusion_model.eval()
        feature_model.eval()
        decoder.eval()
        # prompt.eval()

        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            # Set the imagens from dataloader
            img, img_gray, img_color, _ = create_samples(data)            
            #### Images
            ### Gray Image
            input_img = img_gray.to(device)
            ### Ground truth of the Gray Img
            gt_img = img.to(device)
            ### Get the video Key frame
            key_frame = img_color.to(device)

            l = gt_img.shape[0]

            ### Labels to create sample from noise
            # labels = prompt(key_frame)

            ### Encoder (create feature representation)
            gt_out, skips = feature_model(input_img)

            # ### Exctract color from key frame
            # color_feature = color_neck(key_frame)

            # ### Join the Color features with the Encoder out
            # neck_features = torch.cat((gt_out, color_feature.view(-1,args.in_ch//2,8,8)), axis=1)

            ### Diffusion (due the noise version of input and predict)
            # x = diffusion.sample(diffusion_model, n=l, labels=labels, gray_img=img_gray, in_ch=args.in_ch, create_img=False)

            ### Decoder (create the expected sample using denoised feature space)
            # sampled_images = decoder((neck_features, skips))
            sampled_images = decoder((gt_out, skips))

            for img_idx in range(len(sampled_images)):
                if args.rgb:
                    save_images(tensor_2_img(sampled_images[img_idx].unsqueeze(0)), os.path.join(colored_frames_save,  f"{str(count_frame_idx).zfill(5)}.jpg"))
                else:
                    save_images(tensor_lab_2_rgb(sampled_images[img_idx].unsqueeze(0)), os.path.join(colored_frames_save,  f"{str(count_frame_idx).zfill(5)}.jpg"))
                count_frame_idx+=1

    frame_2_video(colored_frames_save, f"{colored_video_path}/{video_name}_colored.mp4", img_start_name=None)

print("Evaluation Finish")