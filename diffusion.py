from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
import torch
import VAE as vae
from ViT import Vit_neck


## Initializing a scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
## Setting number of sampling steps
scheduler.set_timesteps(51)
## Initializing the U-Net model
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cuda")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import transforms as tfms  

    p = r"C:\video_colorization\data\train\mini_DAVIS_val\pigs\00077.jpg"
    img = vae.load_image(p)
    img = tfms.ToTensor()(img).unsqueeze(0) * 2.0 - 1.0   

    latent_img = vae.pil_to_latents(img)

    decoded_img = vae.latents_to_pil(latent_img)

    noise = torch.randn_like(latent_img) # Random noise

    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    
    for c, sampling_step in enumerate(range(0,51,10)):
        encoded_and_noised = scheduler.add_noise(latent_img, noise, timesteps=torch.tensor([scheduler.timesteps[sampling_step]]))
        axs[c//3][c%3].imshow(vae.latents_to_pil(encoded_and_noised)[0])
        axs[c//3][c%3].set_title(f"Step - {sampling_step}")

    encoded_and_noised = scheduler.add_noise(latent_img, noise, timesteps=torch.tensor([scheduler.timesteps[40]])) ,vae.latents_to_pil(encoded_and_noised)[0]


    import torch, logging
    ## disable warnings
    logging.disable(logging.WARNING)  

    img_encoder = Vit_neck().to("cuda")
    img_encoddings = img_encoder(img.to("cuda")).float().half()

    ## Using U-Net to predict noise    
    latent_model_input = encoded_and_noised[0].to("cuda").float().half()
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,10,encoder_hidden_states=img_encoddings
        )["sample"]
    ## Visualize after subtracting noise 
    out_pred = vae.latents_to_pil(encoded_and_noised[0] - noise_pred)[0]