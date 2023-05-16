import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import kornia as K
from tqdm import tqdm
from torch import optim
from utils import *
from torch.utils.tensorboard import SummaryWriter
import logging
from ViT import *

"""
Code to train a VGG network to predict a similar feature vector
to images in YUV color space. Once the original pre trained for Pytorch
wait for RGB images
"""

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class VGG_19_YUV(nn.Module):
    def __init__(self, out_size=1024, img_size=128) -> None:
        super().__init__()

        # Define the size of image to create last layer size (flatten)
        self.img_size = img_size

        # Load pre-trained VGG network
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # Remove last layer (the classification layer)
        self.vgg_features = torch.nn.Sequential(*list(self.vgg.features.children())[:-1])

        self.out = nn.Sequential(
            nn.Linear(int((img_size**2)*2), out_size*2),
            nn.Linear(out_size*2, out_size)
        )

    def forward(self, x):
        x = self.vgg_features(x)
        return x

# transform=transforms.Compose([
#                 K.color.rgb_to_yuv,
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#             ])

### Training
import read_data as ld
model = "VIT"

model_name = get_model_time()
epochs = 201
device = "cuda"
lr = 2e-3
run_name = f"{model}_{model_name}"
out_size = 1024

image_size=224
batch_size=16

# vgg_vanila = VGG_19(out_size=out_size, img_size=image_size).to(device)
# vgg_vanila.eval()

if model == "VGG":
    vgg_yuv = VGG_19_YUV(out_size=out_size, img_size=image_size).to(device)
else:
    vgg_yuv = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=out_size).to(device)

vgg_yuv.to(device)
vgg_yuv.train()

##### Read Data
dataLoader = ld.ReadData()
used_dataset = "mini_DAVIS"
dataroot = f"C:/video_colorization/data/train/{used_dataset}"
# dataroot = r"C:\video_colorization\data\train\mini_DAVIS"
dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False)

optimizer = optim.AdamW(vgg_yuv.parameters(), lr=lr)

# Define cosine similarity loss function
cosine_sim_loss = nn.CosineEmbeddingLoss()
# Loss to compare the class Images

if model == "VGG":
    metric = nn.MSELoss()
else:
    metric = nn.TripletMarginLoss(margin=1.0, p=2)

metric = nn.MSELoss()

# Log info
logger = SummaryWriter(os.path.join("runs", run_name))
l = len(dataloader)

# def tensor_2_rgb(x):
#     img_rgb = (x.clamp(-1, 1) + 1) / 2
#     img_rgb = tensor_lab_2_rgb(img_rgb).to(device)   
#     img_rgb = (img_rgb.type(torch.float) / 127.5) - 1.0

#     return img_rgb

setup_logging(run_name)

def train():
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            img, img_gray, img_color, _ = create_samples(data)

            # Set imgs to device
            img_yuv = img.to(device)
            img_example = img_color.to(device)

            # y = torch.ones(img_yuv.shape[0]).to(device)

            # Create the vectos of inputs
            vgg_yuv.eval()
            anchor = vgg_yuv(img_example)
            # negative_class = vgg_yuv(negative_img)

            vgg_yuv.train()
            out_yuv = vgg_yuv(img_yuv)

            # Valid loss for each model type
            # if model == "VGG":
            #     loss = metric(out_yuv, anchor)
            # else:
            #     # loss = metric(anchor, out_yuv, negative_class)
            #     loss = cosine_sim_loss(out_yuv, anchor, y)
            #     loss += cosine_sim_loss(out_yuv, negative_class, y*-1)

            loss = metric(out_yuv, anchor)

            # Step in the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(Los=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("epochs", epoch, global_step=epoch * l + i)

            if epoch % 20 == 0:
                os.makedirs(os.path.join("models_vit", run_name), exist_ok=True)
                torch.save(vgg_yuv.state_dict(), os.path.join("models_vit", run_name, f"vit_constrative.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models_vit", run_name, f"vit_optim.pt"))


if __name__ == "__main__":
    train()
    # svd_model_name = "VGG_20230227_223822"
    # saved_model = os.path.join("models", svd_model_name, f"ckpt.pt")
    # model_wights = torch.load(saved_model)
    # vgg_yuv.load_state_dict(model_wights)
    # vgg_yuv.eval()

    # data = next(iter(dataloader))

    # img, img_color, _, _, constrative = create_samples(data, constrative=True)
    # out = vgg_yuv(img)

    print("Done")