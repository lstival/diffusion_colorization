import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

# prompt = "a photo of a bus on the road"
prompt = "albino capybara ultra realistic image"
images = pipe(prompt)  
image = images.images[0]
    
image.save("capybara.png")