import torch
import torch.nn as nn
import kornia as K
from tqdm import tqdm
from torch import optim
from utils import *
from torch.utils.tensorboard import SummaryWriter
import logging
from ViT import *
import read_data as ld

def train():

    # Load the model
    vit = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=out_size).to(device)
    vit.train()

    # Read the data
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False)

    # Define the optimizer and loss function
    optimizer = optim.AdamW(vit.parameters(), lr=lr)
    optimizer

    metric = nn.MSELoss()
    metric.to(device)

    # Log info
    logger = SummaryWriter(os.path.join("runs", run_name))
    l = len(dataloader)

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            img, _, img_color, _ = create_samples(data)

            # Set imgs to device
            img_yuv = img.to(device)
            img_example = img_color.to(device)

            # Create the vectos of inputs
            vit.eval()
            anchor = vit(img_example)

            vit.train()
            out_vit = vit(img_yuv)

            loss = metric(out_vit, anchor)

            # Step in the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(Los=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("epochs", epoch, global_step=epoch * l + i)

        if epoch % 10 == 0:
            os.makedirs(os.path.join("models_vit", run_name), exist_ok=True)
            torch.save(vit.state_dict(), os.path.join("models_vit", run_name, f"vit_constrative.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models_vit", run_name, f"vit_optim.pt"))

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

    device = "cuda"
    model = "VIT"
    model_name = get_model_time()
    run_name = f"{model}_{model_name}"

    epochs = 201
    lr = 2e-3
    out_size = 1024
    image_size=128
    batch_size=350

    used_dataset = "mini_kinetics"
    dataroot = f"C:/video_colorization/data/train/{used_dataset}"

    train()