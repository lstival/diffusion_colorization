import os
from utils import *
import shutil

# Path to the colorized frames
str_date= "DDPM_20230218_090502"
root_colorized_path = r"C:\video_colorization\diffusion\evals"
video_name = "rallye_DAVIS"

# Path to save th video
colored_video_path = os.path.join("videos", str_date, video_name)
os.makedirs(colored_video_path, exist_ok=True)

colorized_path = os.path.join(root_colorized_path, str_date, video_name)

#### List of imagens present in the colorizated imgs

# Name of imgs in the folder
img_in_folder = os.listdir(colorized_path)

# list with all colorized imgs
lst_colorized_images = []

for img_name in img_in_folder:
    if img_name.startswith("out"):
        lst_colorized_images.append(img_name)


# Convert the frames to a video and save
frame_2_video(colorized_path, f"{colored_video_path}/{video_name}_colored.mp4", img_start_name="out")
