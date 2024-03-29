import torch
import os
from utils import *
from ddpm import *
import read_data as ld

# Load the Network to increase the colorization
import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import Vit_neck
import torchvision.models as models
from u_net import *

def train_difussion(pretained_name="UNET_20230323_163628", vit_neck=False ,test_diffusion=False, only_difussion=False):

    ### Load dataset 
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=True)

    ### Models

    ### Vit neck
    if vit_neck:
        ### Encoder and decoder models
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
        diffusion = Diffusion(img_size=8, device=device, noise_steps=noise_steps)
        diffusion_model = UNet_conditional(img_size=8, c_in=in_ch, c_out=in_ch, time_dim=time_dim).to(device)
        # diffusion_model = Reverse_diffusion(c_in=in_ch, c_out=in_ch, time_dim=time_dim, img_size=8).to(device)
        diffusion_model.train()

    elif only_difussion:
        ### Encoder and decoder models
        # feature_model = Encoder(c_in=in_ch, c_out=in_ch, return_subresults=True, img_size=image_size).to(device)
        # feature_model = load_trained_weights(feature_model, pretained_name, "feature")
        # feature_model.eval()

        # decoder = Decoder(c_in=in_ch, c_out=3, img_size=image_size).to(device)
        # decoder.train()

        ### Diffusion process
        diffusion = Diffusion(img_size=image_size, device=device, noise_steps=noise_steps)
        diffusion_model = UNet_conditional(c_in=3, c_out=3, time_dim=time_dim, img_size=image_size).to(device)
        diffusion_model.train()

    else:
        ### Encoder and decoder models
        feature_model = Encoder(c_in=3, c_out=in_ch//2, return_subresults=True, img_size=image_size).to(device)
        feature_model = load_trained_weights(feature_model, pretained_name, "feature")
        feature_model.eval()

        decoder = Decoder(c_in=in_ch, c_out=3, img_size=image_size).to(device)
        decoder = load_trained_weights(decoder, pretained_name, "decoder")
        decoder.eval()

        ### Diffusion process
        diffusion = Diffusion(img_size=image_size//8, device=device, noise_steps=noise_steps)
        # diffusion_model = Reverse_diffusion(c_in=in_ch//2, c_out=in_ch//2, time_dim=time_dim, img_size=8).to(device)
        diffusion_model = UNet_conditional(c_in=in_ch//2, c_out=in_ch//2, time_dim=time_dim, img_size=image_size//8).to(device)
        diffusion_model.train()

    ### Labels generation
    prompt = Vit_neck(batch_size=batch_size, image_size=image_size, out_chanels=time_dim).to(device)
    prompt = load_trained_weights(prompt, vit_name, "vit_constrative", model_path="models_vit")
    prompt.eval()

    ### Optimizers
    mse = nn.MSELoss()
    criterion = SSIMLoss(data_range=1.)
    criterion = criterion.to(device)

    ### Diffusion train
    params_list = diffusion_model.parameters()
    optimizer = optim.AdamW(params_list, lr=lr)

    logger = SummaryWriter(os.path.join("runs", run_name))
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
            
            #### Images

            ### Gray Image
            input_img = img_gray.to(device)
            ### Ground truth of the Gray Img
            gt_img = img.to(device)
            ### Get the video Key frame
            key_frame = img_color.to(device)

            ### Labels to create sample from noise
            labels = prompt(key_frame)

            if vit_neck:
                ### Encoder (create feature representation)
                gt_out, skips = feature_model(input_img)

                ### Exctract color from key frame
                color_feature = color_neck(input_img)

                ### Join the Color features with the Encoder out
                neck_features = torch.cat((gt_out, color_feature.view(-1,in_ch//2,8,8)), axis=1)

                ### Decoder (create the expected sample using denoised feature space)
                # dec_out = decoder((neck_features, skips))

                ### Generate the noise using the ground truth image
                t = diffusion.sample_timesteps(neck_features.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(neck_features, t)

                ### Predict the noise 
                predicted_noise = diffusion_model(x_t, t, labels)

            elif only_difussion:
                ### Generate the noise using the ground truth image
                t = diffusion.sample_timesteps(gt_img.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(gt_img, t)

                ### Predict the noise 
                predicted_noise = diffusion_model(x_t, t, labels)
            else:
                ### Encoder (create feature representation)
                gt_out, skips = feature_model(input_img)

                ### Generate the noise using the ground truth image
                t = diffusion.sample_timesteps(gt_out.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(gt_out, t)

                ### Predict the noise 
                predicted_noise = diffusion_model(x_t, t, labels)

            # ### Meansure the difference between noise predicted and realnoise
            # loss = criterion(tensor_2_img(predicted_noise, int_8=False), tensor_2_img(noise, int_8=False))
            loss = mse(predicted_noise, noise)

            if test_diffusion:
                out_diff = diffusion.sample(diffusion_model, n=l, labels=labels[:l], gray_img=img_gray[:l], in_ch=in_ch, create_img=False)
                out_dec = decoder((out_diff, skips))
                x = (out_dec)
                plot_img = tensor_lab_2_rgb(x)

                loss += mse(x, gt_img)
            
            # loss = mse(x[:,1:], gt_img[:,1:])

            # Gradient Steps and EMA model criation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item(), epoch=epoch)
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("epochs", epoch, global_step=epoch * l + i)

        if epoch % 1000 == 0 and epoch != 0:
            l = 5
            if len(labels) < l:
                l = len(labels)

            # l = len(labels)

            if not test_diffusion and vit_neck:
                x = diffusion.sample(diffusion_model, n=l, labels=labels[:l], in_ch=in_ch, create_img=False)
            elif only_difussion:
                x = diffusion.sample(diffusion_model, n=l, labels=labels[:l], in_ch=3, create_img=False)
            else:
                x = diffusion.sample(diffusion_model, n=len(labels), labels=labels, in_ch=in_ch//2, create_img=False)

            if only_difussion:
                plot_img = x
            else:
                plot_img = decoder((x, skips))
            
            plot_img = tensor_2_img(plot_img)
            
            ### Creating the Folders
            pos_path_save = os.path.join("unet_results", run_name)
            os.makedirs(pos_path_save, exist_ok=True)

            pos_path_save_models = os.path.join("unet_model", run_name)
            os.makedirs(pos_path_save_models, exist_ok=True)

            ### Ploting and saving
            # sampled_images = diffusion.sample(model, n=l, labels=labels[:l], gray_img=img_gray[:l], in_ch=2)
            plot_images(tensor_2_img(gt_img[:l]))
            plot_images(plot_img[:l])
            save_images(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))
            # save the models
            # torch.save(diffusion_model.state_dict(), os.path.join("unet_model", run_name, f"ckpt.pt"))
            # torch.save(feature_model.state_dict(), os.path.join("unet_model", run_name, f"feature.pt"))
            # torch.save(decoder.state_dict(), os.path.join("unet_model", run_name, f"decoder.pt"))
            # torch.save(prompt.state_dict(), os.path.join("unet_model", run_name, f"prompt.pt"))
            # if vit_neck:
            #     torch.save(color_neck.state_dict(), os.path.join("unet_model", run_name, f"vit_neck.pt"))

if __name__ == "__main__":
    model_name = get_model_time()
    run_name = f"UNET_{model_name}"
    epochs = 5001
    noise_steps = 1000


    device="cuda"
    lr=2e-3
    time_dim=1024
    # dataroot = r"C:\video_colorization\data\train\COCO_val2017"
    # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # dataroot = r"C:\video_colorization\data\train\mini_DAVIS"
    # dataroot = r"C:\video_colorization\data\train\rallye_DAVIS"
    dataroot = r"C:\video_colorization\data\train\few_kinetics"

    image_size=128
    in_ch=256
    batch_size=24

    # vit_name = "VIT_20230425_130530"
    vit_name = "VIT_20230426_222111"
    # Train model
        # # # train_difussion("UNET_20230330_132559", vit_neck=True, test_diffusion=False, only_difussion=False)
    # train(vit_neck=False)
    # train_difussion("UNET_20230425_102212", vit_neck=False, test_diffusion=False, only_difussion=True)
    train_difussion("UNET_20230425_152759", vit_neck=False, test_diffusion=False, only_difussion=True)

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
