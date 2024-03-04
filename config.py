import argparse

parser = argparse.ArgumentParser(description="Run Stable Diffusion Diff Edit with command line arguments.")
parser.add_argument("--img_url", '-i', type=str, required=True, help="URL of the image to edit.")
parser.add_argument("--target_prompt", '-t', type=str, required=True, help="Prompt for the target image.")
parser.add_argument("--source_prompt", '-p', type=str, default=None, help="Optional prompt for the source image.")
parser.add_argument("--save_path", '-s', type=str, default=None, help="Optional path to save the edited image.")
parser.add_argument("--device", '-d', type=str, default="cuda", help="Device to run the model on.")
    
args = parser.parse_args()
FLAGS = args