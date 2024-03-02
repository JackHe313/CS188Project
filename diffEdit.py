import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionDiffEditPipeline, DDIMScheduler, DDIMInverseScheduler


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"

init_image = download_image(img_url).resize((768, 768))

pipe = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

mask_prompt = "A bowl of fruits"
prompt = "A bowl of Apples"

mask_image = pipe.generate_mask(image=init_image, source_prompt=prompt, target_prompt=mask_prompt)
image_latents = pipe.invert(image=init_image, prompt=mask_prompt).latents
image = pipe(prompt=prompt, mask_image=mask_image, image_latents=image_latents).images[0]

image.show()