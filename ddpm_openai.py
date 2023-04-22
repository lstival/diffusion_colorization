# import torch
# from torch import autocast
# from diffusers import StableDiffusionPipeline

# # model_id = "runwayml/stable-diffusion-v1-5"
# # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# # pipe = pipe.to("cuda")

# # # prompt = "a photo of a bus on the road"
# # prompt = "albino capybara ultra realistic image"
# # images = pipe(prompt)  
# # image = images.images[0]
    
# # image.save("capybara.png")

# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "dreamlike-art/dreamlike-photoreal-2.0"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens"
# image = pipe(prompt).images[0]



# Vit feature exctration 
# https://huggingface.co/google/vit-base-patch16-224-in21k

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler


# ## Pre treinaed loads
# # 1. Load the autoencoder model which will be used to decode the latents into image space. 
# vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',)

# # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")



# ### Image processing
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# ## Prepare the image input to the Vit Feature Exctration
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
# ## Latent Space of the Image 
last_hidden_states = outputs.last_hidden_state

# ### Diffusion Process
# from diffusers import LMSDiscreteScheduler

# scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# ### Move models to GPU
torch_device = "cuda"
# vae.to(torch_device)
# text_encoder.to(torch_device)
model.to(torch_device)

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
unet.to(torch_device) 