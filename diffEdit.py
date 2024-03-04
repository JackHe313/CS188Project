import PIL
import requests
import torch
from config import FLAGS
import os
from io import BytesIO
from diffusers import StableDiffusionDiffEditPipeline, DDIMScheduler, DDIMInverseScheduler
from transformers import BlipForConditionalGeneration, BlipProcessor, AutoTokenizer, T5ForConditionalGeneration

def download_image(url):
    response = requests.get(url)
    try:
        image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
    except:
        image = PIL.Image.open(BytesIO(requests.get('https://as2.ftcdn.net/v2/jpg/02/25/32/07/1000_F_225320764_OBm2Xby6soDooECWQv25GTtRLzNaSL6g.jpg').content)).convert("RGB")
        print ("Image has error, using default image")    
    return image

@torch.no_grad()
def generate_caption(images, caption_generator, caption_processor):
    text = "a photograph of"

    inputs = caption_processor(images, text, return_tensors="pt").to(device=FLAGS.device, dtype=caption_generator.dtype)
    caption_generator.to(FLAGS.device)
    outputs = caption_generator.generate(**inputs, max_new_tokens=128)

    # offload caption generator
    caption_generator.to("cpu")

    caption = caption_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption

def edit(img_url, target_prompt, source_prompt, save_path):
    init_image = download_image(img_url).resize((768, 768))

    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
    )
    pipe = pipe.to(FLAGS.device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    if source_prompt is not None:
        caption = source_prompt
    else:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        caption = generate_caption(init_image, model, processor)
    print(f"Caption: {caption}")
    mask_image = pipe.generate_mask(image=init_image, source_prompt=caption, target_prompt=target_prompt)
    image_latents = pipe.invert(image=init_image, prompt=caption).latents
    image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents=image_latents).images[0]


    image.show()
    if save_path is not None:
        #check if the path is valid
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        image.save(save_path)

if __name__ == "__main__":
    
    
    MAX_GENRATION_ITERATION = 128
    count = 0

    #while (count < MAX_GENRATION_ITERATION):

    edit(FLAGS.img_url, FLAGS.target_prompt, FLAGS.source_prompt, FLAGS.save_path)
