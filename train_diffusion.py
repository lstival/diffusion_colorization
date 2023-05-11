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
from u_net import Encoder, Decoder
from torchvision.models import vit_b_16

def valid_model(diffusion, diffusion_model, prompt, dataroot, criterion, epoch, logger, l, i):
    ### Load dataset 
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=True)

    with torch.no_grad():
        data = next(iter(dataloader))
        img, img_gray, key_frame, _ = create_samples(data)

        gt_img = img.to(device)
        val_labels = prompt(key_frame)
        input_img = img_gray.to(device)

        latens = vae.pil_to_latents(input_img)

        val_t = diffusion.sample_timesteps(latens.shape[0]).to(device)
        val_x_t, val_noise = diffusion.noise_images(latens, val_t)

        ### Predict the noise 
        predicted_noise = diffusion_model(val_x_t, val_t, val_labels).half()

        ### Meansure the difference between noise predicted and realnoise
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
        dataLoader = ld.ReadData()
        dataloader = dataLoader.create_dataLoader(self.dataroot, image_size, batch_size, shuffle=True)
        return dataloader

    def load_losses(self, mse=True):
        if mse:
            criterion = nn.MSELoss()
        # else:
        #     criterion = SSIMLoss(data_range=1.)

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

        # Load Vit prompt
        ## Labels generation
        # prompt = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=time_dim).to(device)
        # prompt = load_trained_weights(prompt, vit_name, "vit_constrative", model_path="models_vit")
        # prompt = vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1").to("cuda")
        prompt = Vit_neck().to("cuda")
        prompt.eval()

        ## Load Models
        ### Encoder and decoder models
        # feature_model = Encoder(c_in=3, c_out=out_ch//2, return_subresults=True, img_size=image_size).to(device)
        # feature_model = load_trained_weights(feature_model, pretained_name, "feature")
        # feature_model.eval()

        # decoder = Decoder(c_in=out_ch, c_out=3, img_size=image_size).to(device)
        # decoder = load_trained_weights(decoder, pretained_name, "decoder")
        # decoder.eval()

        ### Diffusion process
        diffusion = Diffusion(img_size=image_size//8, device=device, noise_steps=noise_steps)
        diffusion_model = UNet_conditional(c_in=4, c_out=4, time_dim=time_dim, img_size=image_size//8,net_dimension=net_dimension).to(device)
        # diffusion_model = UNet_conditional(c_in=out_ch//2, c_out=out_ch//2, time_dim=time_dim, img_size=image_size//8, net_dimension=net_dimension).to(device)
        diffusion_model.train()

        params_list = diffusion_model.parameters()
        optimizer = optim.Adam(params_list, lr=lr)
        # optimizer = torch.optim.SGD(params_list, lr=lr, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs,  eta_min = 1e-4)
        # lmbda = lambda epoch: 0.95
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

        # Turn on benchmarking for free speed
        torch.backends.cudnn.benchmark = True
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
                # key_frame = img_color.to(input_img)

                ## Labels to create sample from noise
                labels = prompt(input_img)

                ## Encoder (create feature representation)
                latens = vae.pil_to_latents(gt_img)
                # gt_latens = vae.pil_to_latents(gt_img)

                # gt_out, skips = feature_model(gt_img)

                ### Generate the noise using the ground truth image
                t = diffusion.sample_timesteps(latens.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(latens, t)
                # gt_x_t, gt_noise = diffusion.noise_images(gt_latens, t)

                ### Predict the noise 
                predicted_noise = diffusion_model(x_t, t, labels).half()

                ### Meansure the difference between noise predicted and realnoise
                # with torch.autocast(device_type=device):
                loss = criterion(predicted_noise, noise)

                # val_loss, val_data = valid_model(diffusion, diffusion_model, prompt, valid_dataroot, criterion, epoch, logger, l, i)

                # Gradient Steps
                # optimizer.zero_grad()
                for param in diffusion_model.parameters():
                    param.grad = None
                loss.backward()
                optimizer.step()
                pbar.set_postfix(MSE=loss.item(), lr=optimizer.param_groups[0]['lr'], epochs=epoch)
                logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)
                # logger.add_scalar("Val_Loss", val_loss.item(), global_step=epoch * l + i)
            
            scheduler.step()

            if epoch % 10 == 0 and epoch != 0:
                l = 5
                if (gt_img.shape[0]) < l:
                    l = (gt_img.shape[0])

                # x = diffusion.sample(diffusion_model, labels=labels, n=l, in_ch=out_ch//2, create_img=False)
                x = diffusion.sample(diffusion_model, labels=labels[:l], n=l, in_ch=4, create_img=False).half()
                # plot_img = decoder((x, skips))
                # plot_img = tensor_2_img(plot_img)
                plot_img = vae.latents_to_pil(x)

                ### Plot Validation
                # x = diffusion.sample(diffusion_model, labels=val_data[0], n=val_data[0].shape[0], in_ch=out_ch//2, create_img=False)
                # x = diffusion.sample(diffusion_model, labels=val_data[0], n=val_data[0].shape[0], in_ch=4, create_img=False).half()
                # val_plot_img = vae.latents_to_pil(x)
                # val_plot_img = decoder((x, val_data[1]))
                # val_plot_img = tensor_2_img(val_plot_img)
                
                ### Creating the Folders
                pos_path_save = os.path.join("unet_results", run_name)
                os.makedirs(pos_path_save, exist_ok=True)

                pos_path_save_models = os.path.join("unet_model", run_name)
                os.makedirs(pos_path_save_models, exist_ok=True)

                ### Ploting and saving the images
                # plot_images(tensor_2_img(gt_img[:l]))
                plot_images_2(vae.latents_to_pil(latens[:l]))
                plot_images_2(plot_img[:l])
                # plot_images_2(val_plot_img[:l])
                save_images_2(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))
                # Save the models
                torch.save(diffusion_model.state_dict(), os.path.join("unet_model", run_name, f"ckpt.pt"))
                # torch.save(feature_model.state_dict(), os.path.join("unet_model", run_name, f"feature.pt"))
                # torch.save(decoder.state_dict(), os.path.join("unet_model", run_name, f"decoder.pt"))
                # torch.save(prompt.state_dict(), os.path.join("unet_model", run_name, f"prompt.pt"))

        logger.close()

if __name__ == "__main__":
    seed = 2023
    torch.manual_seed(seed)

    model_name = get_model_time()
    run_name = f"UNET_d_{model_name}"
    noise_steps = 80
    time_dim=768
    lr=2e-4
    device="cuda"
    image_size=224
    batch_size=10
    in_ch=256
    out_ch=256
    
    # # dataroot = r"C:\video_colorization\data\train\COCO_val2017"
    # # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # # dataroot = r"C:\video_colorization\data\train\rallye_DAVIS"

    # vit_name = "VIT_20230429_131814"
    pretained_name = "UNET_20230502_130014"
    dataroot = r"C:\video_colorization\data\train\mini_DAVIS"
    valid_dataroot = r"C:\video_colorization\data\train\drone_DAVIS"
    epochs = 501
    net_dimension=128

    training = TrainDiffusion(dataroot, image_size, time_dim)
    training.train(epochs, lr)

    print("Done")