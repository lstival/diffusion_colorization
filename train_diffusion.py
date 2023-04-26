import torch
import os
from utils import *
from ddpm import Diffusion, UNet_conditional
import read_data as ld
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from torch import optim

# Load the Network to increase the colorization
import torch
import torch.nn as nn
# import torch.nn.functional as F
from ViT import Vit_neck
# import torchvision.models as models
from u_net import Encoder, Decoder

class DiffusionModels():
    def __init__(self, rever_diffusion=False) -> None:
        self.rever_diffusion = rever_diffusion

    def train_vit_neck(self, in_ch, c_out, noise_steps, time_dim, pretained_name):
        """
        IF vit neck is true, return the instance of Encoder and Decoder pre trained.
        Also load the Vitneck trained during the encoder and Decoder train.

        Valid if rever diffusion is True, else return the classical U_net to reverse Diffusion.
        The return is a tuple where all models are present in the order:

        feature_model, decoder, color_neck, diffusion, diffusion_model

        """
        feature_model = Encoder(c_in=3, c_out=in_ch//2, return_subresults=True, img_size=image_size).to(device)
        feature_model = load_trained_weights(feature_model, pretained_name, "feature")
        feature_model.eval()

        decoder = Decoder(c_in=in_ch, c_out=3, img_size=image_size, vit_neck=True).to(device)
        decoder = load_trained_weights(decoder, pretained_name, "decoder")
        decoder.eval()

        color_neck = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=int((in_ch//2)*8*8))
        color_neck = load_trained_weights(color_neck, pretained_name, "vit_neck")
        color_neck.eval()

        ### Diffusion process
        diffusion = Diffusion(img_size=image_size//8, device=device, noise_steps=noise_steps)
        diffusion_model = UNet_conditional(c_in=in_ch//2, c_out=in_ch//2, time_dim=time_dim, img_size=image_size//8).to(device)
        diffusion_model.train()

        ### Labels generation
        prompt = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=time_dim).to(device)
        prompt = load_trained_weights(prompt, vit_name, "vit_constrative", model_path="models_vit")
        prompt.eval()

        return (feature_model, decoder, color_neck, diffusion, diffusion_model)

    def train_only_diffusion(self, noise_steps, time_dim):
        """
        Create the diffusion model responsable to create the noise,
        and create the reverse network
        """
        diffusion = Diffusion(img_size=image_size, device=device, noise_steps=noise_steps)
        diffusion_model = UNet_conditional(c_in=3, c_out=3, time_dim=time_dim, img_size=image_size).to(device)
        diffusion_model.train()

        return (diffusion, diffusion_model)

    def train_default(self, pretained_name, noise_steps, time_dim, image_size):
        """
        Create the diffusion model to be trained, using the Encoder and Decaoder pre trained.
        Return a tuple contaning all models
        """
        ### Encoder and decoder models
        feature_model = Encoder(c_in=3, c_out=in_ch//2, return_subresults=True, img_size=image_size).to(device)
        feature_model = load_trained_weights(feature_model, pretained_name, "feature")
        feature_model.eval()

        decoder = Decoder(c_in=in_ch, c_out=3, img_size=image_size).to(device)
        decoder = load_trained_weights(decoder, pretained_name, "decoder")
        decoder.train()

        ### Diffusion process
        diffusion = Diffusion(img_size=image_size//8, device=device, noise_steps=noise_steps)
        diffusion_model = UNet_conditional(c_in=in_ch//2, c_out=in_ch//2, time_dim=time_dim, img_size=image_size//8).to(device)
        diffusion_model.train()

        return (feature_model, decoder, diffusion, diffusion_model)


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
        dataLoader = ld.ReadData()
        dataloader = dataLoader.create_dataLoader(self.dataroot, image_size, batch_size, shuffle=True)
        return dataloader

    def load_losses(self, mse=True):
        if mse:
            criterion = nn.MSELoss()
        else:
            criterion = SSIMLoss(data_range=1.)

        criterion = criterion.to(device)
        return criterion
    
    def train(self, epochs, lr):
        """
        Method to train the reverse diffusion model and the prompt
        """
        criterion = self.load_losses()

        logger = SummaryWriter(os.path.join("runs", self.run_name))

        ## Load Dataset
        dataloader = self.read_datalaoder()

        ## Load Vit prompt
        ### Labels generation
        prompt = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=time_dim).to(device)
        prompt = load_trained_weights(prompt, vit_name, "vit_constrative", model_path="models_vit")
        prompt.eval()

        ## Load Models
        ### Encoder and decoder models
        feature_model = Encoder(c_in=3, c_out=in_ch//2, return_subresults=True, img_size=image_size).to(device)
        feature_model = load_trained_weights(feature_model, pretained_name, "feature")
        feature_model.eval()

        decoder = Decoder(c_in=in_ch, c_out=3, img_size=image_size).to(device)
        decoder = load_trained_weights(decoder, pretained_name, "decoder")
        decoder.eval()

        ### Diffusion process
        diffusion = Diffusion(img_size=image_size//8, device=device, noise_steps=noise_steps)
        diffusion_model = UNet_conditional(c_in=in_ch//2, c_out=in_ch//2, time_dim=time_dim, img_size=image_size//8).to(device)
        diffusion_model.train()

        params_list = diffusion_model.parameters()
        optimizer = optim.AdamW(params_list, lr=lr)

        ## Train Loop
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            for i, (data) in enumerate(pbar):
                ### Get the data and set the samples
                img, img_gray, img_color, _ = create_samples(data)

                ### Avoid break during the sample of denoising
                l = img.shape[0]

                ### Gray Image
                input_img = img_gray.to(device)
                ### Ground truth of the Gray Img
                gt_img = img.to(device)
                ### Get the video Key frame
                key_frame = img_color.to(device)

                ### Labels to create sample from noise
                labels = prompt(key_frame)

                ### Encoder (create feature representation)
                gt_out, skips = feature_model(input_img)

                ### Generate the noise using the ground truth image
                t = diffusion.sample_timesteps(gt_out.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(gt_out, t)

                ### Predict the noise 
                predicted_noise = diffusion_model(x_t, t, labels)

                ### Meansure the difference between noise predicted and realnoise
                loss = criterion(predicted_noise, noise)

                # Gradient Steps and EMA model criation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(MSE=loss.item(), epochs=epoch)
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

            if epoch % 10 == 0:
                l = 5
                if len(labels) < l:
                    l = len(labels)

                x = diffusion.sample(diffusion_model, n=len(labels), labels=labels, in_ch=in_ch//2, create_img=False)

                plot_img = decoder((x, skips))
                
                plot_img = tensor_2_img(plot_img)
                
                ### Creating the Folders
                pos_path_save = os.path.join("unet_results", run_name)
                os.makedirs(pos_path_save, exist_ok=True)

                pos_path_save_models = os.path.join("unet_model", run_name)
                os.makedirs(pos_path_save_models, exist_ok=True)

                ### Ploting and saving the images
                plot_images(tensor_2_img(gt_img[:l]))
                plot_images(plot_img[:l])
                save_images(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))
                # Save the models
                torch.save(diffusion_model.state_dict(), os.path.join("unet_model", run_name, f"ckpt.pt"))
                torch.save(feature_model.state_dict(), os.path.join("unet_model", run_name, f"feature.pt"))
                torch.save(decoder.state_dict(), os.path.join("unet_model", run_name, f"decoder.pt"))
                torch.save(prompt.state_dict(), os.path.join("unet_model", run_name, f"prompt.pt"))

        logger.close()

if __name__ == "__main__":
    model_name = get_model_time()
    run_name = f"UNET_k_{model_name}"
    noise_steps = 1000
    time_dim=1024
    lr=2e-4
    device="cuda"
    image_size=128
    batch_size=100
    in_ch=128
    
    # # dataroot = r"C:\video_colorization\data\train\COCO_val2017"
    # # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # # dataroot = r"C:\video_colorization\data\train\rallye_DAVIS"

    vit_name = "VIT_20230425_130530"
    pretained_name = "UNET_k_20230420_102944"
    dataroot = r"C:\video_colorization\data\train\mini_DAVIS"
    epochs = 201

    training = TrainDiffusion(dataroot, image_size, time_dim)
    training.train(epochs, lr)

    print("Done")