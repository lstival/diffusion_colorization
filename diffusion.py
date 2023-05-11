from diffusers import UNet2DConditionModel, LMSDiscreteScheduler
import torch
import VAE as vae
from ViT import Vit_neck
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


## Initializing a scheduler
noise_setps = 51

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
## Setting number of sampling steps
scheduler.set_timesteps(noise_setps)
## Initializing the U-Net model
# unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cuda")
unet = UNet2DConditionModel(in_channels=4, out_channels=4, norm_num_groups=8,
                            block_out_channels= (128, 256, 512, 512), cross_attention_dim=768).to("cuda")
# unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)

for param in unet.parameters():
    param.requires_grad = False

for param in unet.up_blocks[-1:].parameters():
    param.requires_grad = True

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from torchvision import transforms as tfms  

    p = r"C:\video_colorization\data\train\mini_DAVIS\drone\00077.jpg"
    img = vae.load_image(p)
    imgimg_gray = tfms.Grayscale(num_output_channels=3)(img)
    img = tfms.ToTensor()(img).unsqueeze(0) * 2.0 - 1.0   

    latent_img = vae.pil_to_latents(img)

    decoded_img = vae.latents_to_pil(latent_img)

    noise = torch.randn_like(latent_img) # Random noise

    fig, axs = plt.subplots(2, 3, figsize=(16, 12))
    
    for c, sampling_step in enumerate(range(0,noise_setps,10)):
        encoded_and_noised = scheduler.add_noise(latent_img, noise, timesteps=torch.tensor([scheduler.timesteps[sampling_step]]))
        axs[c//3][c%3].imshow(vae.latents_to_pil(encoded_and_noised)[0])
        axs[c//3][c%3].set_title(f"Step - {sampling_step}")

    encoded_and_noised = scheduler.add_noise(latent_img, noise, timesteps=torch.tensor([scheduler.timesteps[noise_setps-1]])) ,vae.latents_to_pil(encoded_and_noised)[0]

    ### Reference Image
    p_ref = r"C:\video_colorization\data\train\mini_DAVIS\drone\00078.jpg"
    img_ref = vae.load_image(p_ref)
    img_ref = tfms.ToTensor()(img_ref).unsqueeze(0) * 2.0 - 1.0   

    import torch, logging
    ## disable warnings
    logging.disable(logging.WARNING)  

    img_encoder = Vit_neck().to("cuda")
    img_encoddings = img_encoder(img_ref.to("cuda")).float().half()

    ## Using U-Net to predict noise    
    latent_model_input = encoded_and_noised[0].to("cuda").float().half()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            noise_pred = unet(
                latent_model_input, 50, encoder_hidden_states=img_encoddings
            )["sample"]
        
    ## Visualize after subtracting noise 
    out_pred = vae.latents_to_pil(encoded_and_noised[0] - noise_pred)[0]

    plt.imshow(out_pred)
    plt.show()