import torch
from PIL import Image
from torchvision import transforms as tfms  
## Basic libraries 
import numpy as np 

## Loading a VAE model 
from diffusers import AutoencoderKL 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to("cuda")

def load_image(p):
    '''     
    Function to load images from a defined path     
    '''    
    return Image.open(p).convert('RGB').resize((224,224))

def pil_to_latents(image):
    '''     
    Function to convert image to latents     
    '''     
    # init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0   
    init_image = image.to(device="cuda", dtype=torch.float16)
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215     
    return init_latent_dist  

def latents_to_pil(latents):     
    '''     
    Function to convert latents to images     
    '''     
    latents = (1 / 0.18215) * latents     
    with torch.no_grad():         
        image = vae.decode(latents).sample     
    
    image = (image / 2 + 0.5).clamp(0, 1)     
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
    images = (image * 255).round().astype("uint8")     
    pil_images = [Image.fromarray(image) for image in images]        
    return pil_images


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    p = r"C:\video_colorization\data\train\mini_DAVIS\tennis\00058.jpg"
    img = load_image(p)
    init_image = tfms.ToTensor()(img).unsqueeze(0) * 2.0 - 1.0  

    latent_img = pil_to_latents(init_image)

    decoded_img = latents_to_pil(latent_img)
    
    plt.imshow(decoded_img[0])