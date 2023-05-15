import torch
import os
from utils import *
from ddpm import Diffusion, UNet_conditional
import read_data as ld
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from torch import optim
import VAE as vae

# Load the Network to increase the colorization
import torch
import torch.nn as nn
# import torch.nn.functional as F
from ViT import Vit_neck
# import torchvision.models as models


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def valid_model(diffusion, diffusion_model, prompt, dataroot, criterion):
    ### Load dataset 
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=False)

    with torch.no_grad():
        data = next(iter(dataloader))
        img, img_gray, _, _ = create_samples(data)

        gt_img = img.to(device)
        input_img = img_gray.to(device)

        ### Get the prompt
        val_labels = prompt(gt_img)

        ### Get the latents
        latens = vae.pil_to_latents(input_img)

        ### Get the noise
        val_t = diffusion.sample_timesteps(latens.shape[0]).to(device)
        val_x_t, val_noise = diffusion.noise_images(latens, val_t)

        ### Predict the noise 
        predicted_noise = diffusion_model(val_x_t, val_t, val_labels).half()

        ### Meansure the difference between noise predicted and realnoise
        with torch.autocast(device_type=device):
            val_loss = criterion(predicted_noise, val_noise)

    return val_loss, [val_labels]

class TrainDiffusion():
    def __init__(self, dataroot, image_size, time_dim) -> None:

        self.dataroot = dataroot
        self.image_size = image_size
        self.time_dim = time_dim
        self.run_name = get_model_time()

    def read_datalaoder(self):
        """
        Get the data from the dataroot and return the dataloader
        """
        ### Load dataset 
        dataLoader = ld.ReadLatent()
        dataloader = dataLoader.create_dataLoader(self.dataroot, batch_size, shuffle=True)
        return dataloader

    def load_losses(self, mse=True):
        if mse:
            criterion = nn.MSELoss()

        criterion = criterion.to(device)
        return criterion
    
    def train(self, epochs, lr, pretained_name=None):
        """
        Method to train the reverse diffusion model and the prompt
        """
        criterion = self.load_losses()

        logger = SummaryWriter(os.path.join("runs", self.run_name))

        ## Load Dataset
        dataloader = self.read_datalaoder()

        # Load Vit prompt
        prompt = Vit_neck().to("cuda")
        prompt.eval()

        ### Diffusion process
        diffusion = Diffusion(img_size=image_size//8, device=device, noise_steps=noise_steps)
        diffusion_model = UNet_conditional(c_in=4, c_out=4, time_dim=time_dim, img_size=image_size//8,net_dimension=net_dimension).to(device)
        if pretained_name:
            resume(diffusion_model, os.path.join("unet_model", pretained_name, "best_model.pth"))
        diffusion_model.train()

        params_list = diffusion_model.parameters()
        optimizer = optim.Adam(params_list, lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        # Turn on benchmarking for free speed
        torch.backends.cudnn.benchmark = True
        ## Train Loop

        best_loss = 9999

        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            for i, (data) in enumerate(pbar):
                latents, labels, next_frame = data

                ### Move the data to the device
                latents=latents.to(device)
                labels=labels.to(device)

                l = latents.shape[0]

                ### Generate the noise using the ground truth image
                t = diffusion.sample_timesteps(latents.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(latents, t)

                ### Predict the noise 
                predicted_noise = diffusion_model(x_t, t, labels).half()

                ### Meansure the difference between noise predicted and realnoise
                with torch.autocast(device_type=device):
                    loss = criterion(predicted_noise, noise)

                # val_loss, val_data = valid_model(diffusion, diffusion_model, prompt, valid_dataroot, criterion)

                ### Gradient Steps
                optimizer.zero_grad()
                # for param in diffusion_model.parameters():
                    # param.grad = None
                loss.backward()
                optimizer.step()

                ### Update the progress bar
                pbar.set_postfix(MSE=loss.item(), lr=optimizer.param_groups[0]['lr'], epochs=epoch)
                logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)

            # scheduler.step()
            if loss.item() < best_loss:

                ### Create the folder to save the model
                pos_path_save_models = os.path.join("unet_model", run_name)
                os.makedirs(pos_path_save_models, exist_ok=True)

                best_loss = loss.item()
                best_epoch = epoch
                # checkpoint(diffusion_model, os.path.join("unet_model", run_name, "best_model.pth"))

            # elif epoch - best_epoch > early_stop_thresh:
            #     print(f"Early stopping at epoch {epoch}")
            #     break

            # resume(diffusion_model, os.path.join("unet_model", run_name, "best_model.pth"))

            if epoch % 10 == 0 and epoch != 0:
                l = 5
                if (latents.shape[0]) < l:
                    l = (latents.shape[0])

                x = diffusion.sample(diffusion_model, labels=labels[:l], n=l, in_ch=4, create_img=False).half()
                plot_img = vae.latents_to_pil(x)
                
                ### Creating the Folders
                pos_path_save = os.path.join("unet_results", run_name)
                os.makedirs(pos_path_save, exist_ok=True)

                ### Ploting and saving the images
                plot_images_2(vae.latents_to_pil(latents[:l]))
                plot_images_2(plot_img[:l])

                ### Plot Validation
                # x = diffusion.sample(diffusion_model, labels=val_data[0], n=val_data[0].shape[0], in_ch=4, create_img=False).half()
                # val_plot_img = vae.latents_to_pil(x)
                # plot_images_2(val_plot_img[:l])

                save_images_2(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))

                # Save the models
                # if loss.item() < best_loss:
                torch.save(diffusion_model.state_dict(), os.path.join("unet_model", run_name, f"ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join("unet_model", run_name, f"optimizer.pt"))

        logger.close()

if __name__ == "__main__":
    ### Hyperparameters
    seed = 2023
    torch.manual_seed(seed)

    model_name = get_model_time()
    run_name = f"UNET_d_{model_name}"
    noise_steps = 100
    time_dim=1000
    lr=2e-5
    device="cuda"
    image_size=224
    batch_size=40
    early_stop_thresh = 50
    
    # # dataroot = r"C:\video_colorization\data\train\COCO_val2017"
    # # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # # dataroot = r"C:\video_colorization\data\train\rallye_DAVIS"

    # vit_name = "VIT_20230429_131814"
    pretained_name = "UNET_d_20230515_145243"
    used_dataset = "DAVIS"
    dataroot = f"C:/video_colorization/diffusion/data/latens/{used_dataset}/"
    valid_dataroot = r"C:\video_colorization\data\train\mini_DAVIS_val"
    epochs = 501
    net_dimension=220

    training = TrainDiffusion(dataroot, image_size, time_dim)
    training.train(epochs, lr)

    print("Done")