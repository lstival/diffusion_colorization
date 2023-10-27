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
dataset = "LDV"
dataype = "train"
# dataroot = f"C:/video_colorization/data/train/{dataset}"
dataroot = f"C:/video_colorization/data/{dataype}/{dataset}"
latensroot = f"data/latens/{dataset}/"
image_size = 224
batch_size = 1
device = "cuda"
size_data = 1
# latent_filename = "latents_transf"
latent_filename = "latents"

list_latens = []
list_labels = []

for i in range(size_data):
    ### Create the dataset
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False, train=True)
    if i == size_data -1:
        dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False, train=None)

    ### Load models
    prompt = vit.Vit_neck().to(device)

    ### Create the folder to save the latens
    os.makedirs(latensroot, exist_ok=True)

    ## Loop for get the latents

    pbar = tqdm(dataloader)
    for data in pbar:
        pbar.set_description(f" Epoc: {i}, Extracting features")
        img, img_gray, _, _ = create_samples(data)

        latens = vae.pil_to_latents(img)
        list_latens.append(latens.to("cpu").detach().numpy())

        labels = prompt(img_gray)
        list_labels.append(labels.to("cpu").detach().numpy())

np.savez(latensroot+latent_filename, labels=list_labels, latents=list_latens)

del list_labels, list_latens

print("Done!")
data = np.load(latensroot+latent_filename+".npz", 'r')

print("Labels Shape: ", data["labels"].shape)
print("Latents Shape: ", data["latents"].shape)