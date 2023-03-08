import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

#Convert imga to Lab
from PIL import ImageCms

# Spatial size of training images. All images will be resized to this
# image_size = 256

# # Batch size during training
# batch_size = 5

# Root directory for dataset
dataroot = "../../data/train/initial_argument"

#Reference to create class to manipule the colors spaces
# https://pillow.readthedocs.io/en/stable/reference/ImageCms.html

class ColorTrans:

    '''Class for transforming RGB<->LAB color spaces for PIL images.'''
    
    def __init__(self):
        self.srgb_p = ImageCms.createProfile("sRGB")
        self.lab_p  = ImageCms.createProfile("LAB")
        self.rgb2lab_trans = ImageCms.buildTransformFromOpenProfiles(self.srgb_p, self.lab_p, "RGB", "LAB")
        self.lab2rgb_trans = ImageCms.buildTransformFromOpenProfiles(self.lab_p, self.srgb_p, "LAB", "RGB")
    
    def rgb2lab(self, img):
        return ImageCms.applyTransform(img, self.rgb2lab_trans)

    def lab2rgb(self, img):
        return ImageCms.applyTransform(img, self.lab2rgb_trans)

#Class to manipulet the color space of the images
color_trans = ColorTrans()

# Create the dataset
class ReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, batch_size, shuffle=True):
        """
        Read all imagens from the folder and convert them in lab space,
        as result a Data Loader as created and returned:

        dataroot: Path to the images
        Image_size: Dimension of the images (Default 512,768)
        shuffle: Define if the samples will be ordem randomized (Default: True)
        batch_size: Number the samples for batch (Default: 2)

        """
        dataset = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        # transforms.CenterCrop(image_size),
                                        #convert RGB image to LAB
                                        # transforms.Lambda(lambda x: color_trans.rgb2lab(x)),
                                        # transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        # Alexnet
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]))
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=shuffle)
        return self.dataloader

    def img_example(self, dataloader):
        """
        Return a random sample present in the DataLoader
        """
        img = next(iter(dataloader))
        tensor2image = transforms.ToPILImage()
        exmp_img = tensor2image(img[0][0])
        return exmp_img
