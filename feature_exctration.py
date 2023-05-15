import torch
import pickle
import VAE as vae
import read_data as ld
import ViT as vit
from utils import *
import os
from tqdm import tqdm
import numpy as np

"""
This script is for extract the features from the images and save it in a file,
this process was realized to save time during the diffusion training.
"""

### Parameters
dataset = "DAVIS"
dataroot = f"C:/video_colorization/data/train/{dataset}"
latensroot = f"data/latens/{dataset}/"
image_size = 224
batch_size = 1
device = "cuda"

### Create the dataset
dataLoader = ld.ReadData()
dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False)

### Load models
vit = vit.Vit_neck().to(device)

### Create the folder to save the latens
os.makedirs(latensroot, exist_ok=True)

list_latens = []
list_labels = []

## Loop for get the latents
pbar = tqdm(dataloader)
for data in pbar:
    pbar.set_description("Extracting features")
    img, img_gray, _, _ = create_samples(data)

    latens = vae.pil_to_latents(img)
    list_latens.append(latens.to("cpu").detach().numpy())

    labels = vit(img_gray)
    list_labels.append(labels.to("cpu").detach().numpy())

np.savez(latensroot+"latent.npz", labels=list_labels, latents=list_latens)

del list_labels, list_latens

print("Done!")
data = np.load(latensroot+"latent.npz", 'r')

print("Labels Shape: ", data["labels"].shape)
print("Latents Shape: ", data["latents"].shape)