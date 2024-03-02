import PIL
import requests
import torch
import argparse
from io import BytesIO
from diffusers import StableDiffusionDiffEditPipeline, DDIMScheduler, DDIMInverseScheduler

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

def edit(img_url, target_prompt, source_prompt):
    init_image = download_image(img_url).resize((768, 768))

    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    mask_image = pipe.generate_mask(image=init_image, source_prompt=source_prompt, target_prompt=target_prompt)
    image_latents = pipe.invert(image=init_image, prompt=source_prompt).latents
    image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents=image_latents).images[0]

    image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable Diffusion Diff Edit with command line arguments.")
    parser.add_argument("--img_url", '-i', type=str, required=True, help="URL of the image to edit.")
    parser.add_argument("--target_prompt", '-t', type=str, required=True, help="Prompt for the target image.")
    parser.add_argument("--source_prompt", '-s', type=str, required=True, help="Prompt for the source image.")
    
    args = parser.parse_args()
    
    edit(args.img_url, args.target_prompt, args.source_prompt)
