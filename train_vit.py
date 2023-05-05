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
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False, constrative=True)
    val_dataloader = dataLoader.create_dataLoader(val_dataroot, image_size, batch_size, shuffle=False, constrative=True)

    # Define the optimizer and loss function
    optimizer = optim.AdamW(vit.parameters(), lr=lr)

    # metric = nn.MSELoss()
    # metric = nn.TripletMarginLoss(margin=1.0, p=2)
    # metric = nn.CosineEmbeddingLoss(margin=1)
    # metric = nn.CrossEntropyLoss()
    # metric = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(dim=1, eps=1e-6))
    metric = (nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y)))
    metric.to(device)

    # Log info
    logger = SummaryWriter(os.path.join("runs", run_name))
    l = len(dataloader)
    vit.train()

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            img, _, img_color, _, img_random = create_samples(data)
            # img, img_color, next_frame, img_random = data

            # Set imgs to device
            anchor = img.to(device)
            positive = img_color.to(device)
            negative = img_random.to(device)

            # Create the vectos of inputs
            positive_out = vit(positive)
            negative_out = vit(negative)
            anchor_out = vit(anchor)

            # labels = torch.ones(anchor_out.shape[0]).to(device)

            loss = metric(anchor_out, positive_out, negative_out)
            # loss = metric(out_vit, anchor)
            # loss = metric(anchor_out, positive_out, negative_out)
            # pos_loss = metric(anchor_out, positive_out, labels)
            # neg_loss = metric(anchor_out, negative_out, labels*-1)

            # loss = metric(anchor_out, positive_out)

            vit.eval()
            with torch.no_grad():
                val_data = next(iter(val_dataloader))
                val_img, _, val_img_color, val_next_frame, val_img_random = create_samples(val_data)

                # Set imgs to device
                val_anchor = val_img.to(device)
                val_positive = val_img_color.to(device)
                val_negative = val_img_random.to(device)

                # Create the vectos of inputs
                val_positive_out = vit(val_positive)
                val_negative_out = vit(val_negative)
                val_anchor_out = vit(val_anchor)

                # labels = torch.ones(anchor_out.shape[0]).to(device)

                val_loss = metric(val_anchor_out, val_positive_out, val_negative_out)
            vit.train()

            # Step in the gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(Los=loss.item(), Val_loss=val_loss.item(), epochs=epoch)
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)

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

    epochs = 501
    lr = 2e-3
    out_size = 3072
    image_size=128
    batch_size=500

    used_dataset = "mini_DAVIS"
    dataroot = f"C:/video_colorization/data/train/{used_dataset}"
    val_dataroot = f"C:/video_colorization/data/train/drone_DAVIS"

    dataLoader = ld.ReadData()
    

    train()
    # dataLoader = ld.ReadData()
    # dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False, constrative=True)