import torch
import os
from utils import *
import read_data as ld

# Load the Network to increase the colorization
import torch
import torch.nn as nn
from u_net import *

def valid_model(encoder, decoder, dataroot, criterion, criterion_2, epoch, logger, l, i, creater_img=False):
    ### Load dataset 
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=True)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        data = next(iter(dataloader))
        img, img_gray, _, _ = create_samples(data)

        input_img = img_gray.to(device)
        gt_img = img.to(device)

        gt_out, skips = encoder(input_img)
        dec_out = decoder((gt_out, skips))

        val_loss = criterion_2(dec_out, gt_img)   
        val_loss += criterion(tensor_2_img(dec_out, int_8=False), tensor_2_img(gt_img, int_8=False))

        # logger.add_scalar("MSE Val", loss.item(), global_step=epoch * l + i)
    encoder.train()
    decoder.train()

    if creater_img:
        plot_img = tensor_2_img(dec_out[:l])
        plot_images(plot_img)
    else:
        return val_loss


def train():
    logger = SummaryWriter(os.path.join("runs", run_name))
    ### Load dataset 
    dataLoader = ld.ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, image_size, batch_size, shuffle=True)

    ### Encoder and decoder models
    feature_model = Encoder(c_in=3, c_out=out_ch//2, return_subresults=True, img_size=image_size).to(device)
    feature_model.train()

    decoder = Decoder(c_in=out_ch, c_out=3, img_size=image_size).to(device)
    decoder.train()

    params_list = list(decoder.parameters()) + list(feature_model.parameters())

    ### Optimizers
    criterion = SSIMLoss(data_range=1.)
    criterion_2 = nn.MSELoss()
    
    criterion.to(device)
    criterion_2.to(device)

    ### Encoder / Decoder train
    optimizer = optim.AdamW(params_list, lr=lr)

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
            img, img_gray, _, _ = create_samples(data)

            ### Avoid break during the sample of denoising
            l = img.shape[0]
            
            ### Select thw 2 imagens to training
            input_img = img_gray.to(device) # Gray Image
            gt_img = img.to(device) # Ground truth of the Gray Img   

            ### Encoder (create feature representation)
            gt_out, skips = feature_model(input_img)
            
            dec_out = decoder((gt_out, skips))

            x = dec_out

            loss = criterion_2(x, gt_img)            
            loss += criterion(tensor_2_img(x, int_8=False), tensor_2_img(gt_img, int_8=False))

            val_loss = valid_model(feature_model, decoder, valid_dataroot, criterion, criterion_2, epoch, logger, l, i, creater_img=False)

            # Gradient Steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(MSE=loss.item(), Val_loss=val_loss.item(), epochs=epoch)
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            logger.add_scalar("epochs", epoch, global_step=epoch * l + i)


        if epoch % 10 == 0:
            l = 5

            ### Creating the Folders
            pos_path_save = os.path.join("unet_results", run_name)
            os.makedirs(pos_path_save, exist_ok=True)

            pos_path_save_models = os.path.join("unet_model", run_name)
            os.makedirs(pos_path_save_models, exist_ok=True)

            ### Ploting and saving
            plot_images(tensor_2_img(gt_img[:l]))

            plot_img = tensor_2_img(x[:l])
            plot_images(plot_img)

            ### Valid Model in other dataset
            valid_model(feature_model, decoder, valid_dataroot, criterion, criterion_2, epoch, logger, l, i, creater_img=True)
            
            save_images(plot_img, os.path.join("unet_results", run_name, f"{epoch}.jpg"))
            torch.save(feature_model.state_dict(), os.path.join("unet_model", run_name, f"feature.pt"))
            torch.save(decoder.state_dict(), os.path.join("unet_model", run_name, f"decoder.pt"))
            torch.save(optimizer.state_dict(), os.path.join("unet_model", run_name, f"optim.pt"))

if __name__ == "__main__":
    model_name = get_model_time()
    run_name = f"UNET_{model_name}"
    epochs = 201
    
    batch_size=16
    image_size=128
    in_ch=256
    out_ch = 768

    device="cuda"
    lr=2e-3
    time_dim=1024
    # dataroot = r"C:\video_colorization\data\train\COCO_val2017"
    # dataroot = r"C:\video_colorization\data\train\mini_DAVIS"
    # dataroot = r"C:\video_colorization\data\train\mini_kinetics"
    # dataroot = r"C:\video_colorization\data\train\rallye_DAVIS"

    # dataroot = r"C:\video_colorization\data\train\kinetics_5per"
    dataroot = r"C:\video_colorization\data\train\mini_DAVIS"
    valid_dataroot = r"C:\video_colorization\data\train\mini_DAVIS_val"
    
    # Train model
    train()
    print("Done")