import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import kornia as K
from utils import *
import random

# path = "C:/video_colorization/data/train/DAVIS"
# path_gray = "C:/video_colorization/Vit-autoencoder/temp"

class ConstrativeDataset(Dataset):
    def __init__(self, path, image_size) -> None:
        super(Dataset, self).__init__()
        self.path = path
        self.image_size = image_size

        self.scenes = os.listdir(path)

        self.color_examples, self.samples, self.lst_rand_scenes = self.__getscenes__(self.path)

    def __getscenes__(self, path):
        """
        Return two list: One with the samples the other
        with the first frame in the folder
        """
        color_examples = []
        samples = []
        lst_rand_scenes = []

        for scene in self.scenes:
            # Select the random scene
            rand_scene = self.__randon_scene__(scene)
            # Parch to the video folders
            scene_path = os.path.join(path, scene)

            # Path to the random selected scene
            rand_scene_path = os.path.join(self.path, rand_scene)

            for scene_frame in range(len(os.listdir(scene_path))-1):
                color_examples.append(self.__transform__(Image.open(f"{scene_path}/{str(0).zfill(5)}.jpg"))) #Img 0
                samples.append(self.__transform__(Image.open(f"{scene_path}/{str(scene_frame+1).zfill(5)}.jpg"))) # Other Imgs
                lst_rand_scenes.append(self.__transform__(Image.open(os.path.join(rand_scene_path, f"{str(random.randrange(0,10)).zfill(5)}.jpg" ))))

        assert len(samples) == len(color_examples)

        return samples, color_examples, lst_rand_scenes
    
    def __randon_scene__(self, scene):
        """
        Return a scene difference of the scene input name, this process
        is important to garant the triplet training (create embedding more
        close to the same scene, and at same time distante of the others
        classes)
        """

        # Select a random scene
        rand_scene = random.choice(self.scenes)

        # Loop to change the random scene while is equal the actual scene
        while rand_scene == scene:
            rand_scene = random.choice(self.scenes)
        
        return rand_scene

    def __transform__(self, x):
        """
        Recives a sample of PIL images and return
        they normalized and converted to a tensor.
        """

        transform=transforms.Compose([
                        # torchvision.transforms.Resize(150),  # args.image_size + 1/4 *args.image_size
                        torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                        transforms.Resize((self.image_size,self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        # K.color.RgbToYuv(),
                    ])

        x_transformed = transform(x)
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
        # Try get the next frame, if cant possible get the previous one.
        try:
            next_frame = self.samples[index+1]
        except:
            next_frame = self.samples[index-1]

        return self.color_examples[index], self.samples[index], next_frame, self.lst_rand_scenes[index]
    
class PosProcessingDataset(Dataset):
    def __init__(self, path, image_size, pos_path) -> None:
        super(Dataset, self).__init__()

        self.path = path
        self.image_size = image_size
        self.pos_path = pos_path

        self.scenes = os.listdir(path)

        self.color_examples, self.samples, self.pos_color = self.__getscenes__(self.path)

    def __getscenes__(self, path):
        """
        Return two list: One with the samples the other
        with the first frame in the folder
        """
        color_examples = []
        samples = []
        pos_color = []

        for scene in self.scenes:
            # Parch to the video folders
            scene_path = os.path.join(path, scene)
            # Patch to frames created by the trained network
            pos_scene_path = os.path.join(self.pos_path, scene+'.mp4')

            for scene_frame in range(len(os.listdir(scene_path))-1):
                key_frame = random.randint(0, 10)
                color_examples.append(self.__transform__(Image.open(f"{scene_path}/{str(key_frame).zfill(5)}.jpg"))) #Img 0
                samples.append(self.__transform__(Image.open(f"{scene_path}/{str(scene_frame+1).zfill(5)}.jpg"))) # Other Imgs
                pos_color.append(self.__transform__(Image.open(f"{pos_scene_path}/{str(scene_frame).zfill(5)}.jpg"))) # Colorized

        assert len(samples) == len(color_examples)

        return samples, color_examples, pos_color

    def __transform__(self, x):
        """
        Recives a sample of PIL images and return
        they normalized and converted to a tensor.
        """
    
        transform=transforms.Compose([
                        torchvision.transforms.Resize(160),  # args.image_size + 1/4 *args.image_size
                        torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                        transforms.Resize((self.image_size,self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        # K.color.RgbToYuv(),
                    ])

        x_transformed = transform(x)
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
        # Try get the next frame, if cant possible get the previous one.
        try:
            next_frame = self.samples[index+1]
        except:
            next_frame = self.samples[index-1]

        return self.color_examples[index], self.samples[index], next_frame, self.pos_color[index]

class DAVISDataset(Dataset):
    """
    This class create a dataset from a path, where the data must be
    divided in subfoldes for each scene or video.
    The return is a list with 3 elements, frames to ve colorized,
    the direclety next frames in the video sequence and example image with color.
    """

    def __init__(self, path, image_size, rgb=True, pos_path=None, constrative=None, DAVIS=None):
        super(Dataset, self).__init__()

        self.path = path
        self.rgb = rgb
        self.DAVIS = DAVIS
        self.image_size = image_size

        self.scenes = os.listdir(path)

        self.color_examples, self.samples = self.__getscenes__(self.path)

    def __getscenes__(self, path):
        """
        Return two list: One with the samples the other
        with the first frame in the folder
        """
        color_examples = []
        samples = []

        if self.DAVIS:
            for scene in self.scenes:
                # Parch to the video folders
                scene_path = os.path.join(path, scene)

                for scene_frame in range(len(os.listdir(scene_path))-1):
                    key_frame = random.randint(0, 10)
                    key_frame = 10
                    color_examples.append(self.__transform__(Image.open(f"{scene_path}/{str(key_frame).zfill(5)}.jpg"))) #Img 0
                    samples.append(self.__transform__(Image.open(f"{scene_path}/{str(scene_frame+1).zfill(5)}.jpg"))) # Other Imgs
        else:
            for scene in self.scenes:
            # Parch to the video folders
                scene_path = os.path.join(path, scene)

                for scene_frame in (os.listdir(scene_path)):
                    key_frame = (os.listdir(scene_path))[random.randint(0, 10)]
                    color_examples.append(self.__transform__(Image.open(f"{scene_path}/{str(key_frame)}").convert('RGB'))) #Img 0
                    samples.append(self.__transform__(Image.open(f"{scene_path}/{str(scene_frame)}").convert('RGB'))) # Other Imgs

        assert len(samples) == len(color_examples)

        return samples, color_examples

    def __transform__(self, x):
        """
        Recives a sample of PIL images and return
        they normalized and converted to a tensor.
        """
        transform=transforms.Compose([
            # torchvision.transforms.Resize(160),  # args.image_size + 1/4 *args.image_size
            torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
            transforms.Resize((self.image_size,self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        x_transformed = transform(x)
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
        # Try get the next frame, if cant possible get the previous one.
        try:
            next_frame = self.samples[index+1]
        except:
            next_frame = self.samples[index-1]

        return self.color_examples[index], self.samples[index], next_frame


# Create the dataset
class ReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size=16, shuffle=False, rgb=True, pos_path=None, constrative=None, DAVIS=True):

        if pos_path is not None:
            self.datas = PosProcessingDataset(dataroot, image_size, pos_path)
        elif constrative is not None:
            self.datas = ConstrativeDataset(dataroot, image_size)
        else:
            self.datas = DAVISDataset(dataroot, image_size, DAVIS)

        # self.datas = DAVISDataset(dataroot, image_size, rgb=rgb, pos_path=pos_path, constrative=constrative)
        self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle)

        assert (next(iter(self.dataloader))[0].shape) == (next(iter(self.dataloader))[1].shape), "The shapes must be the same"
        return self.dataloader


if __name__ == '__main__':
    print("main")

    image_size = 128
    batch_size = 4

    dataLoader = ReadData()
    date_str = "DDPM_20230218_090502"
    used_dataset = "mini_DAVIS"

    dataroot = f"C:/video_colorization/data/train/{used_dataset}"
    pos_dataroot = os.path.join("C:/video_colorization/diffusion/evals", date_str, used_dataset)

    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False, constrative=False, DAVIS=False)

    data = next(iter(dataloader))

    # img, img_gray, img_color, next_frame, pos_frame = create_samples(data, pos_imgs=True)
    img, img_gray, img_color, next_frame, constrastive = create_samples(data, pos_imgs=False)

    # Plt the ground truth img
    b = tensor_lab_2_rgb(img)
    plot_images(b)

    # Plt gray img
    b_gray = tensor_2_img(img_gray)
    plot_images(b_gray)

    # Plot color img (example)
    b_color = tensor_lab_2_rgb(img_color)
    plot_images(b_color)

    # Plot other class img
    img_constrastive = tensor_lab_2_rgb(constrastive)
    plot_images(img_constrastive)

    # # Plot color img (example)
    # c_color = tensor_lab_2_rgb(pos_frame)
    # plot_images(c_color)


# from PIL import Image

# root_path = "C:/video_colorization/data/train/DAVIS/rallye"
# x = Image.open(f"{root_path}/00043.jpg")
# x2 = Image.open(f"{root_path}/00010.jpg")

# transform=transforms.Compose([
#                             transforms.Resize((128, 128)),
#                             # transforms.ToTensor(),
#                             # K.color.rgb_to_lab,
#                             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                             # K.enhance.Normalize(mean=torch.zeros(3), std=torch.ones(3)),
#                             # K.enhance.normalize_min_max,
#                             # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#                         ])
# x_transformed = transform(x)
# x2_transformed = transform(x2)

# x_transformed.convert('L').save("00042_128_gray.jpg")
# x2_transformed.save("00010_128.jpg")


# Loop for each scene to get the 

# cls = ReadData()
# temp_path = "temp/images"
# dataloader = cls.create_dataLoader(temp_path.split('/')[0], (128, 128), 1)

# datas = DAVISDataset(path, (256, 256))

# dataloader = torch.utils.data.DataLoader(datas, batch_size=16,shuffle=False)
# print(next(iter(dataloader))[0].shape)
# print(next(iter(dataloader))[1].shape)
# # Save the first image of each sample
# img = next(iter(dataloader))[0]
# img_color = next(iter(dataloader))[1]
# next_frame = next(iter(dataloader))[2]
# # Plot min max
# print(img.max())
# print(img.min())

# import matplotlib.pyplot as plt
# plt.imshow(img[0].transpose(0,2))
