import PIL
from PIL import Image
import requests
import torch
from config import FLAGS
import os
import cv2
import numpy as np
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

def return_mask(init_image, target_prompt, source_prompt):
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
    return mask_image, caption, pipe


def edit(target_prompt, mask_image, init_image, caption, pipe,save_path):
    
    image_latents = pipe.invert(image=init_image, prompt=caption).latents
    image = pipe(prompt=target_prompt, mask_image=mask_image, image_latents=image_latents).images[0]
    #convert numpy array to PIL image

    image.show()
    if save_path is not None:
        #check if the path is valid
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        image.save(save_path)
    return image, mask_image

if __name__ == "__main__":
    
    
    MAX_GENRATION_ITERATION = 128
    count = 0
    init_image = download_image(FLAGS.img_url).resize((768, 768))

    while (count < MAX_GENRATION_ITERATION):
        
        mask_image, caption, pipe = return_mask(init_image, FLAGS.target_prompt, FLAGS.source_prompt)
        
        init_image = np.array(init_image)
        mask = cv2.resize(mask_image, (init_image.shape[1], init_image.shape[0]))

        # Create a red color mask
        colored_mask = np.zeros_like(init_image)
        colored_mask[mask > 0] = [0, 0, 255]  # BGR format, so red is [0, 0, 255]

        # Create semi-transparent background (dimming effect)
        background = np.zeros_like(init_image, dtype=np.uint8)
        background[:] = [0, 0, 0]  # Black background
        alpha = 0.5  # Transparency for the non-masked area
        semi_transparent_bg = cv2.addWeighted(init_image, alpha, background, 1 - alpha, 0)

        # Overlay the red mask on the semi-transparent background
        # Masked area will be red, and the rest will be semi-transparent
        final_image = np.where(colored_mask > 0, colored_mask, semi_transparent_bg)

        # Convert back to RGB if needed (OpenCV uses BGR)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

        # Show or save your final image
        cv2.imshow('Final Image', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('Is there any change you want to make to the mask?')
        print('If yes, type "y"')
        print('If no, type "n"')
        change = input()
        if (change == 'y'):
            count += 1
            continue
        elif (change == 'n'):
            break
        else:
            print('Invalid input, please try again')
            break


    while (count < MAX_GENRATION_ITERATION):    
        print('Are you satisfied with the result?')
        edit(FLAGS.target_prompt, mask_image, init_image, caption, pipe, FLAGS.save_path)
        satisfied = input()
        if (satisfied == 'y'):
            break
        elif (satisfied == 'n'):
            count += 1
            continue
        else:
            print('Invalid input, please try again')
            break
