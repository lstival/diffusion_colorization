import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import kornia as K
from utils import *
import random
import numpy as np

class ColorizationDataset(Dataset):
    """
    This class create a dataset from a path, where the data must be
    divided in subfoldes for each scene or video.
    The return is a list with 3 elements, frames to ve colorized,
    the direclety next frames in the video sequence and example image with color.
    """

    def __init__(self, path, image_size, constrative=False):
        super(Dataset, self).__init__()

        self.path = path
        self.image_size = image_size
        self.constrative = constrative

        self.scenes = os.listdir(path)

        self.dataset = torchvision.datasets.ImageFolder(self.path, self.__transform__)

    def __colorization_transform__(self, x):

        colorization_transform=transforms.Compose([
                torchvision.transforms.Resize(280),  # args.image_size + 1/4 *args.image_size
                torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        return colorization_transform(x)

    def __transform__(self, x):
        """
        Recives a sample of PIL images and return
        they normalized and converted to a tensor.
        """

        x_transformed = self.__colorization_transform__(x)
        return x_transformed
        
    def __len__(self):
        """
        Return hou much samples as in the dataset.
        """
        return len(self.dataset)

    
    def __getitem__(self, index):
        """
        Return the frames that will be colorized, the next frames and 
        the color example frame (first of the sequence).
        """
        # Get the next indices
        keyframe_index = random.randint(0,10)
        next_index = min(index + 1, len(self.dataset) - 1)
        
        if self.constrative:
            random_idx = random.randint(20, 100)
            random_idx = min(random_idx + 1, len(self.dataset) - 1)
            return self.dataset[index], self.dataset[keyframe_index], self.dataset[next_index], self.dataset[random_idx] 
        else:
            return self.dataset[index], self.dataset[keyframe_index], self.dataset[next_index]
        

class LatentsDataset(Dataset):
    """
    This class create a dataset from a path, where the data must be where
    the latents excracted from the images are saved.
    And return 3 elements, the latents, the next latents and the labels.
    Where latents are the features extracted from the images and labels are
    the exctration of VAC pre trained.
    """

    def __init__(self, path):
        super(Dataset, self).__init__()

        self.path = path

        self.dataset = np.load(path+"latent.npz", 'r')
        ## Latents of the images
        self.latents = self.dataset['latents']
        ## Labels of the images (img_gray)
        self.labels = self.dataset['labels']

        
    def __len__(self):
        """
        Return hou much samples as in the dataset.
        """
        return len(self.latents)

    
    def __getitem__(self, index):
        """
        Return the frames that will be colorized, the next frames and 
        the color example frame (first of the sequence).
        """
        # Get the next indices
        last_index = min(index + 1, len(self.latents) - 1)
        return self.latents[index].squeeze(), self.labels[index].squeeze(), self.latents[last_index].squeeze()

# Create the dataset
class ReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size=16, shuffle=False, pin_memory=True, constrative=False):

        self.datas = ColorizationDataset(dataroot, image_size, constrative=constrative)

        # self.datas = DAVISDataset(dataroot, image_size, rgb=rgb, pos_path=pos_path, constrative=constrative)
        self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

        # assert (next(iter(self.dataloader))[0][0].shape) == (next(iter(self.dataloader))[1].shape), "The shapes must be the same"
        return self.dataloader
    
class ReadLatent():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, batch_size, shuffle=False, pin_memory=True):

        self.datas = LatentsDataset(dataroot)

        # self.datas = DAVISDataset(dataroot, image_size, rgb=rgb, pos_path=pos_path, constrative=constrative)
        self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

        # assert (next(iter(self.dataloader))[0][0].shape) == (next(iter(self.dataloader))[1].shape), "The shapes must be the same"
        return self.dataloader

if __name__ == '__main__':
    print("main")

    image_size = 224
    batch_size = 4
    used_dataset = "mini_DAVIS"
    dataroot = f"C:/video_colorization/diffusion/data/latens/{used_dataset}/"

    dataLoader = ReadLatent()
    dataloader = dataLoader.create_dataLoader(dataroot, batch_size, shuffle=True)

    data = next(iter(dataloader))

    latent_gt, latent_gray, latent_next = data

    print(latent_gt.shape)
    print(latent_gray.shape)
    print(latent_next.shape)

    import VAE as vae
    import matplotlib.pyplot as plt

    out_img = vae.latents_to_pil(latent_gt.to("cuda"))
    next_out_img = vae.latents_to_pil(latent_next.to("cuda"))

    plt.imshow(out_img[0])
    plt.imshow(next_out_img[0])

    # dataLoader = ReadData()
    # date_str = "DDPM_20230218_090502"

    # pos_dataroot = os.path.join("C:/video_colorization/diffusion/evals", date_str, used_dataset)

    # dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=True)

    # data = next(iter(dataloader))

    # # img, img_gray, img_color, next_frame, pos_frame = create_samples(data, pos_imgs=True)
    # img, img_gray, img_color, next_frame, = create_samples(data, pos_imgs=False)

    # # Plt the ground truth img
    # b = tensor_2_img(img)
    # plot_images(b)

    # # Plt gray img
    # b_gray = tensor_2_img(img_gray)
    # plot_images(b_gray)

    # # Plot color img (example)
    # b_color = tensor_2_img(img_color)
    # plot_images(b_color)

    # # Plot color img (example)
    # c_color = tensor_2_img(next_frame)
    # plot_images(c_color)