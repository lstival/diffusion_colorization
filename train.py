import torch
import os
from utils import *
from ddpm import *
import DAVIS_dataset as ld

# Load the Network to increase the colorization
import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import Vit_neck
import torchvision.models as models
from u_net import *

def train():
    ### Load dataset 
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=True, rgb=False)

    ### Models
    
    ### Encoder and decoder models
    feature_model = Encoder(c_in=3, c_out=in_ch, return_subresults=True, img_size=image_size).to(device)
    feature_model.train()

    decoder = Decoder(c_in=in_ch*2, c_out=3, img_size=image_size).to(device)
    decoder.train()

    ### Optimizers
    mse = nn.MSELoss()

    ### Encoder / Decoder train
    params_list = list(decoder.parameters()) + list(feature_model.parameters())
    optimizer = optim.AdamW(params_list, lr=lr)

    best_loss = 9999

    ### Train Loop
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            """
            ### Encoder / Decoder train
            Encoder recives the gray scale image and genereate a feature space
            Decoder get the Encoder output and create the output image
            Calculate the loss between the GT image (img) and the Decoder output
            """
            img, img_gray, img_color, next_frame = create_samples(data)

            ### Avoid break during the sample of denoising
            l = img.shape[0]
            
            ### Select thw 2 imagens to training
            input_img = img_gray.to(device) # Gray Image
            gt_img = img.to(device) # Ground truth of the Gray Img   

            ### Encoder (create feature representation)
            gt_out, skips = feature_model(input_img)

            ### Decoder (create the expected sample using denoised feature space)
            dec_out = decoder((gt_out, skips))

            # dec_out = decoder((gt_out, skips))
            x = dec_out
            # y = scale_0_and_1(img_gray[:,:1])
            # x = torch.cat([y, dec_out],1)

            loss = mse(x, gt_img)
            # loss = mse(x[:,1:], gt_img[:,1:])

            # Gradient Steps and EMA model criation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            l = 5

            plot_img = tensor_lab_2_rgb(x[:l])
            ### Creating the Folders
            pos_path_save = os.path.join("unet_results", run_name)
            os.makedirs(pos_path_save, exist_ok=True)

            pos_path_save_models = os.path.join("unet_model", run_name)
            os.makedirs(pos_path_save_models, exist_ok=True)

            ### Ploting and saving
            plot_images(tensor_lab_2_rgb(gt_img[:l]))
            plot_images(plot_img)
            save_images(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))
            torch.save(feature_model.state_dict(), os.path.join("unet_model", run_name, f"feature.pt"))
            torch.save(decoder.state_dict(), os.path.join("unet_model", run_name, f"decoder.pt"))


def train_difussion(pretained_name="UNET_20230323_163628"):
    ### Load dataset 
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=True, rgb=False)

    ### Models
    
    ### Encoder and decoder models
    feature_model = Encoder(c_in=3, c_out=in_ch, return_subresults=True, img_size=image_size).to(device)
    feature_model = load_trained_weights(feature_model, pretained_name, "feature")
    feature_model.eval()

    decoder = Decoder(c_in=in_ch*2, c_out=3, img_size=image_size).to(device)
    decoder = load_trained_weights(decoder, pretained_name, "decoder")
    decoder.eval()

    ### Diffusion process
    diffusion = Diffusion(img_size=image_size, device=device, noise_steps=noise_steps)
    diffusion_model = UNet_conditional(c_in=in_ch, c_out=in_ch, time_dim=time_dim).to(device)
    diffusion_model.train()

    ### Labels generation
    prompt = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=time_dim)
    prompt.train()

    ### Optimizers
    mse = nn.MSELoss()

    ### Diffusion train
    params_list = list(prompt.parameters()) + list(diffusion_model.parameters())
    optimizer = optim.AdamW(params_list, lr=lr)

    best_loss = 9999

    ### Train Loop
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (data) in enumerate(pbar):
            """
            ### First part (Diffusion training)

            Prompt get the key frame colorized and exctract the features (from ViT)
            Encoder recives the gray scale image and genereate a feature space
            Create the noise schedule and the noise version of the input (output of the decoder)
            Predict the noise for the select noise step
            Calculate the loss for noise prediction (meansure the quality in recreat the Encoder output)


            ### Second Part (Image comparation)
            Make the reverse process of diffusion (denoising the randon noise in the Encoder output)
            Decoder get the diffusion output and skip connections from Encoder (to generate the colorized Frame)
            Calculate the loss between the GT image (img) and the Decoder output
            """
            img, img_gray, img_color, next_frame = create_samples(data)

            ### Avoid break during the sample of denoising
            l = img.shape[0]
            
            ### Select thw 2 imagens to training
            input_img = img_gray.to(device) # Gray Image
            gt_img = img.to(device) # Ground truth of the Gray Img   
            color_example_img = img_color.to(device) # Key frame with color examples

            ### Labels to create sample from noise
            labels = prompt(color_example_img)

            ### Encoder (create feature representation)
            gt_out, skips = feature_model(input_img)

            ### Generate the noise using the ground truth image
            t = diffusion.sample_timesteps(gt_out.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(gt_out, t)

            # ### Predict the noise 
            predicted_noise = diffusion_model(x_t, t, labels)

            # ### Meansure the difference between noise predicted and realnoise
            loss = mse(predicted_noise, noise)

            # ### Diffusion (due the noise version of input and predict)
            dif_out = diffusion.sample(diffusion_model, n=l, labels=labels, gray_img=img_gray, in_ch=in_ch, create_img=False)

            ### Decoder (create the expected sample using denoised feature space)
            dec_out = decoder((dif_out, skips))

            # dec_out = decoder((gt_out, skips))
            x = dec_out
            # y = scale_0_and_1(img_gray[:,:1])
            # x = torch.cat([y, dec_out],1)

            loss = mse(x, gt_img)
            # loss = mse(x[:,1:], gt_img[:,1:])

            # Gradient Steps and EMA model criation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            l = 5
            if len(labels) < l:
                l = len(labels)

            plot_img = tensor_lab_2_rgb(x[:l])
            ### Creating the Folders
            pos_path_save = os.path.join("unet_results", run_name)
            os.makedirs(pos_path_save, exist_ok=True)

            pos_path_save_models = os.path.join("unet_model", run_name)
            os.makedirs(pos_path_save_models, exist_ok=True)

            ### Ploting and saving
            # sampled_images = diffusion.sample(model, n=l, labels=labels[:l], gray_img=img_gray[:l], in_ch=2)
            plot_images(tensor_lab_2_rgb(gt_img[:l]))
            plot_images(plot_img)
            save_images(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))
            torch.save(diffusion_model.state_dict(), os.path.join("unet_model", run_name, f"ckpt.pt"))
            torch.save(feature_model.state_dict(), os.path.join("unet_model", run_name, f"feature.pt"))
            torch.save(decoder.state_dict(), os.path.join("unet_model", run_name, f"decoder.pt"))

if __name__ == "__main__":
    model_name = get_model_time()
    run_name = f"UNET_{model_name}"
    epochs = 101
    noise_steps = 200

    image_size=64
    batch_size=16
    device="cuda"
    lr=2e-3
    time_dim=1024
    dataroot = r"C:\video_colorization\data\train\mini_DAVIS"
    in_ch=128

    # Train model
    train_difussion("UNET_20230323_171416")
    # train()

    # img = torch.ones((batch_size, 3, image_size, image_size)).to(device)

    # feature_mdel = Encoder(c_in=3, return_subresults=True,img_size=image_size).to(device)
    # decoder = Decoder(c_in=256,img_size=image_size).to(device)

    # diffusion = Diffusion(img_size=image_size, device=device, noise_steps=noise_steps)
    # diffusion_model = UNet_conditional(c_in=256, c_out=256, time_dim=time_dim).to(device)
    # diffusion_model.train()

    # out = feature_mdel(img)
    # dec_out = decoder(out)

    print("Done")

    # decoder = Decoder(c_in=512)
    # a = torch.ones([2, 512, 8, 8])
    # decoder(a)

    # x4.shape
    # torch.Size([4, 256, 16, 16])

    # img = torch.ones((4,3,128,128))
    # out = model(img)
