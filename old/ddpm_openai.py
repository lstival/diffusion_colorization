import torch
# # from torch import autocast
# # from diffusers import StableDiffusionPipeline

# # # model_id = "runwayml/stable-diffusion-v1-5"
# # # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# # # pipe = pipe.to("cuda")

# # # # prompt = "a photo of a bus on the road"
# # # prompt = "albino capybara ultra realistic image"
# # # images = pipe(prompt)  
# # # image = images.images[0]
    
# # # image.save("capybara.png")

# # from diffusers import StableDiffusionPipeline
# # import torch

# # model_id = "dreamlike-art/dreamlike-photoreal-2.0"
# # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# # pipe = pipe.to("cuda")

# # prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens"
# # image = pipe(prompt).images[0]



# # Vit feature exctration 
# # https://huggingface.co/google/vit-base-patch16-224-in21k

# from transformers import ViTImageProcessor, ViTModel
# from PIL import Image
# import requests
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler


# # ## Pre treinaed loads
# # # 1. Load the autoencoder model which will be used to decode the latents into image space. 
# # vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# # # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',)

# # # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
# # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")



# # ### Image processing
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

# # ## Prepare the image input to the Vit Feature Exctration
# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# # ## Latent Space of the Image 
# last_hidden_states = outputs.last_hidden_state

# # ### Diffusion Process
# # from diffusers import LMSDiscreteScheduler

# # scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# # ### Move models to GPU
# torch_device = "cuda"
# # vae.to(torch_device)
# # text_encoder.to(torch_device)
# model.to(torch_device)

# # 3. The UNet model for generating the latents.
# unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
# unet.to(torch_device) 



#####
# https://towardsdatascience.com/stable-diffusion-using-hugging-face-501d8dbdd8

## Imaging  library 
from PIL import Image 
from torchvision import transforms as tfms  
## Basic libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import urllib.request

## Loading a VAE model 
from diffusers import AutoencoderKL 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to("cuda")
def load_image(p):
    '''     
    Function to load images from a defined path     
    '''    
    return Image.open(p).convert('RGB').resize((512,512))

def pil_to_latents(image):
    '''     
    Function to convert image to latents     
    '''     
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0   
    init_image = init_image.to(device="cuda", dtype=torch.float16)
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

### Load the image
# p = FastDownload().download('https://lafeber.com/pet-birds/wp-content/uploads/2018/06/Scarlet-Macaw-2.jpg')
urllib.request.urlretrieve(
'https://media.geeksforgeeks.org/wp-content/uploads/20210318103632/gfg-300x300.png',
"gfg.png")
# p = "gfg.png"
p = r"C:\video_colorization\data\train\mini_DAVIS_val\pigs\00077.jpg"
img = load_image(p)
print(f"Dimension of this image: {np.array(img).shape}")


### Convert the image to latents

latent_img = pil_to_latents(img)
print(f"Dimension of this latent representation: {latent_img.shape}")

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for c in range(4):
    axs[c].imshow(latent_img[0][c].detach().cpu(), cmap='Greys')

### Convert the latents to image
decoded_img = latents_to_pil(latent_img)

### Plot the image
plt.imshow(decoded_img[0])
plt.show()

