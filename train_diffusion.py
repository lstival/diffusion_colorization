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
import numpy as np 

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

# Load the Network to increase the colorization
import torch
import torch.nn as nn
# import torch.nn.functional as F
from ViT import Vit_neck
# import torchvision.models as models


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

class TrainDiffusion():
    def __init__(self, dataroot, valid_dataroot, image_size, time_dim) -> None:

        self.dataroot = dataroot
        self.image_size = image_size
        self.time_dim = time_dim
        self.run_name = get_model_time()
        self.valid_dataroot = valid_dataroot

    def read_datalaoder(self):
        """
        Get the data from the dataroot and return the dataloader
        """
        ### Load dataset 
        dataLoader = ld.ReadLatent()
        dataloader = dataLoader.create_dataLoader(self.dataroot, batch_size, shuffle=True, valid_dataroot=self.valid_dataroot)
        return dataloader
    
    def read_dataset(self):
        """
        Get the data from the dataroot and return the dataset
        """
        ### Load dataset 
        dataLoader = ld.ReadLatent()
        dataset = dataLoader.create_dataset(self.dataroot)
        val_dataset = dataLoader.create_dataset(self.valid_dataroot)
        return dataset, val_dataset

    def load_losses(self, mse=True):
        if mse:
            criterion = nn.MSELoss()

        criterion = criterion.to(device)
        return criterion
    
    def train_epoch(self, diffusion, diffusion_model, device, dataloader, criterion, optimizer, epoch):
        diffusion_model.train()
        for latents, labels, _ in dataloader:

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train_pbar.update()

        return loss, l, labels, latents
    
    def valid_epoch(self, diffusion, diffusion_model, device, dataloader, criterion, epoch):

        diffusion_model.eval()
        with torch.no_grad():
            for latents, labels, _ in dataloader:

                latents=latents.to(device)
                labels=labels.to(device)

                l = latents.shape[0]

                ### Generate the noise using the ground truth image
                t = diffusion.sample_timesteps(latents.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(latents, t)

                ### Predict the noise 
                predicted_noise = diffusion_model(x_t, t, labels).half()

                with torch.autocast(device_type=device):
                    val_loss = criterion(predicted_noise, noise)

                # val_pbar.update()

        return val_loss, l ,labels
    
    def train(self, epochs, lr, pretained_name=None):
        """
        Method to train the reverse diffusion model and the prompt
        """

        ## Cross validation parameters
        k=10
        splits=KFold(n_splits=k,shuffle=True,random_state=42)

        ## Load Dataset
        # dataloader = self.read_datalaoder()
        dataset, val_dataset = self.read_dataset()

        # Load Vit prompt
        prompt = Vit_neck().to("cuda")
        prompt.eval()

        ### Diffusion process
        diffusion = Diffusion(img_size=image_size//8, device=device, noise_steps=noise_steps)

        # Turn on benchmarking for free speed
        # torch.backends.cudnn.benchmark = True
        best_loss = 999
        diffusion_model = UNet_conditional(c_in=4, c_out=4, time_dim=time_dim, img_size=image_size//8,net_dimension=net_dimension).to(device)

        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    
            ## Read pretrained weights
            if pretained_name:
                # resume(diffusion_model, os.path.join("unet_model", pretained_name, "ckpt.pt"))
                resume(diffusion_model, os.path.join("unet_model", pretained_name, "best_model.pt"))

            params_list = diffusion_model.parameters()
            optimizer = optim.Adam(params_list, lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

            # print(f'Fold {fold + 1}')

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)

            ## Train dataloader
            train_loader = DataLoader(dataset, batch_size=batch_size,  sampler=train_sampler)
            # train_pbar = tqdm(train_loader, desc="Training", leave=False)

            ## Validation dataloader
            val_loader = DataLoader(dataset, batch_size=batch_size,sampler=test_sampler)
            # val_pbar = tqdm(val_loader, desc="Validation", leave=False)

            criterion = self.load_losses()

            logger = SummaryWriter(os.path.join("runs", self.run_name))

            ### Loop over the epochs
            epoch_pbar = tqdm(range(epochs), desc="Epochs", leave=True)
            for epoch in epoch_pbar:
                logging.info(f"Starting epoch {epoch}:")

                ## Pbar setings
                #   
                ## Train diffusion model
                loss, l, labels, latents = self.train_epoch(diffusion, diffusion_model, device, train_loader, criterion, optimizer, epoch)
                
                ## Evaluate diffusion model
                val_loss, val_l, val_labels = self.valid_epoch(diffusion, diffusion_model, device, val_loader, criterion, epoch)

                ### Update the logger
                logger.add_scalar("Loss", loss.item(), global_step=epoch * l)

                epoch_pbar.set_postfix(MSE=loss.item(), MSE_val=val_loss.item(), lr=optimizer.param_groups[0]['lr'],  best_loss=best_loss)
                # epoch_pbar.reset()

                # scheduler.step()

                test_loss = loss.item()
                if test_loss < best_loss:

                    ### Create the folder to save the model
                    pos_path_save_models = os.path.join("unet_model", run_name)
                    os.makedirs(pos_path_save_models, exist_ok=True)

                    best_loss = loss.item()
                    # best_epoch = epoch
                    checkpoint(diffusion_model, os.path.join("unet_model", run_name, "best_model.pt"))
                    torch.save(optimizer.state_dict(), os.path.join("unet_model", run_name, f"best_optimizer.pt"))

                # elif epoch - best_epoch > early_stop_thresh:
                #     print(f"Early stopping at epoch {epoch}")
                #     break

                # resume(diffusion_model, os.path.join("unet_model", run_name, "best_model.pt"))
                # optimizer.load_state_dict(torch.load(os.path.join("unet_model", run_name, "best_optimizer.pt")))

                if epoch % 10 == 0 and epoch != 0:
                    l = 5
                    if (labels.shape[0]) < l:
                        l = (labels.shape[0])

                    x = diffusion.sample(diffusion_model, labels=labels[:l], n=l, in_ch=4, create_img=False).half()
                    plot_img = vae.latents_to_pil(x)
                    
                    ### Creating the Folders
                    pos_path_save = os.path.join("unet_results", run_name)
                    os.makedirs(pos_path_save, exist_ok=True)

                    ### Ploting and saving the images
                    plot_images_2(vae.latents_to_pil(latents[:l]))
                    plot_images_2(plot_img[:l])

                    ### Plot Validation
                    x = diffusion.sample(diffusion_model, labels=val_labels[:l], n=l, in_ch=4, create_img=False).half()
                    val_plot_img = vae.latents_to_pil(x)
                    plot_images_2(val_plot_img[:l])

                    save_images_2(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))

                    # Save the models
                    # if loss.item() < best_loss:
                    torch.save(diffusion_model.state_dict(), os.path.join("unet_model", run_name, f"ckpt.pt"))
                    torch.save(optimizer.state_dict(), os.path.join("unet_model", run_name, f"optimizer.pt"))

                    nome_arquivo = f"better_loss.txt"
                    arquivo = open(os.path.join("unet_model", run_name, nome_arquivo), "w")
                    arquivo.write(f"Epoch {epoch} - {best_loss}")
                    arquivo.close()

            torch.cuda.empty_cache()
            logger.close()

if __name__ == "__main__":
    ### Hyperparameters
    seed = 42
    torch.manual_seed(seed)

    model_name = get_model_time()
    run_name = f"UNET_d_{model_name}"
    noise_steps = 100
    time_dim=1000
    lr=2e-5
    device="cuda"
    image_size=224
    net_dimension=128
    

    # # dataroot = r"C:\video_colorization\data\train\COCO_val2017"
    # # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # # dataroot = r"C:\video_colorization\data\train\rallye_DAVIS"
    
    pretained_name = "UNET_d_20230518_153443"
    # pretained_name = None
    used_dataset = "mini_kinetics"
    dataroot = f"C:/video_colorization/diffusion/data/latens/{used_dataset}/"
    valid_dataroot = f"C:/video_colorization/diffusion/data/latens/mini_DAVIS/"

    early_stop_thresh = 50
    epochs = 201
    batch_size=60

    training = TrainDiffusion(dataroot, valid_dataroot, image_size, time_dim)
    training.train(epochs, lr, pretained_name)

    print("Done")