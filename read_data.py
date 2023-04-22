import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import kornia as K
from utils import *
import random

class ColorizationDataset(Dataset):
    """
    This class create a dataset from a path, where the data must be
    divided in subfoldes for each scene or video.
    The return is a list with 3 elements, frames to ve colorized,
    the direclety next frames in the video sequence and example image with color.
    """

    def __init__(self, path, image_size):
        super(Dataset, self).__init__()

        self.path = path
        self.image_size = image_size

        self.scenes = os.listdir(path)

        self.color_examples, self.samples = self.__getscenes__(self.path)

    def __colorization_transform__(self, x):

        colorization_transform=transforms.Compose([
                torchvision.transforms.Resize(160),  # args.image_size + 1/4 *args.image_size
                torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        return colorization_transform(x)

    def __getscenes__(self, path):
        """
        Return two lists: One with the samples and the other with the first frame in the folder
        """
        color_examples = []
        samples = []
        
        for scene in self.scenes:
            # Path to the video folder
            scene_path = os.path.join(path, scene)
            
            # Load list of frame names for the scene
            scene_frames = os.listdir(scene_path)
            
            # Get a random selection of key frames for the scene
            key_frame_names = random.sample(scene_frames, len(scene_frames))
            
            # Load sample images and append to list
            for key_idx, scene_frame_name in enumerate(scene_frames):

                # Load color example images and append to list
                key_frame_name = key_frame_names[key_idx]
                key_frame = Image.open(os.path.join(scene_path, key_frame_name))
                color_examples.append(self.__transform__(key_frame))
                sample = Image.open(os.path.join(scene_path, scene_frame_name))
                samples.append(self.__transform__(sample))
        
        assert len(samples) == len(color_examples), "Samples and our color version need the same amount of samples"
        
        return samples, color_examples

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
        return len(self.color_examples)

    
    def __getitem__(self, index):
        """
        Return the frames that will be colorized, the next frames and 
        the color example frame (first of the sequence).
        """
        # Get the next indices
        next_index = min(index + 1, len(self.samples) - 1)
        
        return self.color_examples[index], self.samples[index], self.color_examples[next_index]
    
# Create the dataset
class ReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size=16, shuffle=False):

        self.datas = ColorizationDataset(dataroot, image_size)

        # self.datas = DAVISDataset(dataroot, image_size, rgb=rgb, pos_path=pos_path, constrative=constrative)
        self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle, num_workers=10)

        assert (next(iter(self.dataloader))[0].shape) == (next(iter(self.dataloader))[1].shape), "The shapes must be the same"
        return self.dataloader

if __name__ == '__main__':
    print("main")

    image_size = 128
    batch_size = 4

    dataLoader = ReadData()
    date_str = "DDPM_20230218_090502"
    used_dataset = "drone_DAVIS"

    dataroot = f"C:/video_colorization/data/train/{used_dataset}"
    pos_dataroot = os.path.join("C:/video_colorization/diffusion/evals", date_str, used_dataset)

    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False)

    data = next(iter(dataloader))

    # img, img_gray, img_color, next_frame, pos_frame = create_samples(data, pos_imgs=True)
    img, img_gray, img_color, next_frame, = create_samples(data, pos_imgs=False)

    # Plt the ground truth img
    b = tensor_2_img(img)
    plot_images(b)

    # Plt gray img
    b_gray = tensor_2_img(img_gray)
    plot_images(b_gray)

    # Plot color img (example)
    b_color = tensor_2_img(img_color)
    plot_images(b_color)

    # Plot color img (example)
    c_color = tensor_2_img(next_frame)
    plot_images(c_color)