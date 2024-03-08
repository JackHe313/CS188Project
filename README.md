# Stable Diffusion Diff Edit

### Install dependencies
```
pip install -r requirements.txt 
```
### usage: 
```
Python diffEdit.py [--img_url IMG_URL] --target_prompt TARGET_PROMPT [--source_prompt SOURCE_PROMPT] [--save_path SAVE_PATH] [--device DEVICE] [--seg_prompt SEG_PROMPT] [--seed SEED]
```

### For End-to-End Image Generation Editing:
#### User uploaded Image Editing
Example
```
python diffEdit.py -i "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png" -t "a bowl of pears"
```
In which case the program will generate the mask and ask about your satisfaction about it. 

![Mask](/img/mask.png)

When user accepts the mask, it will edit the image based on the target prompt.

#### User Generate Image Editing
Example
```
python diffEdit.py -p "a bowl of fruits"
```
This time instead of providing the image, we generate the image based on Stable Diffusion first. Given the prompt, it will display a generated image and if user are not satisfied, it will ask for the prompt for changes and do editing as the above process.

### For Text based Object Segmentation
Example
```
python diffEdit.py -i "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png" -sp -t "fruit"
```
It will generate the mask(segmentation) correspond with the target prompt.

### All Options:
```
  -h, --help            show this help message and exit
  --img_url IMG_URL, -i IMG_URL
                        URL of the image to edit.
  --target_prompt TARGET_PROMPT, -t TARGET_PROMPT
                        Prompt for the target image.
  --source_prompt SOURCE_PROMPT, -p SOURCE_PROMPT
                        Optional prompt for the source image.
  --save_path SAVE_PATH, -s SAVE_PATH
                        Optional path to save the edited image.
  --device DEVICE, -d DEVICE
                        Device to run the model on.
  --seg_prompt, -sp     Boolean flag to indicate if the target_prompt is a segment prompt.
  --seed SEED           Seed for the random number generator.
```
