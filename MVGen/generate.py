import os
import torch
import argparse
import yaml
from src.lightning_pano_gen import PanoGenerator
import numpy as np
import cv2
from generate_video_tool.pano_video_generation import generate_video
from PIL import Image
import exiftool
from exiftool import ExifToolHelper
from datetime import datetime

torch.manual_seed(0)

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fov', type=int, default=90, help='fov')
    parser.add_argument('--deg', type=int, default=45, help='degree')
    parser.add_argument('--gen_video', action='store_true', help='generate video')
    parser.add_argument('--save_frames', action='store_true')
    parser.add_argument('--prompt_folder', type=str, default='../prompt/NGGS_example', help='path to prompt folder')
 
    return parser.parse_args()

def generate_for_scene(scene_name, scene_text, args, model, resolution, config, base_dir):
    num = int(360 / args.deg)
    Rs, Ks = [], []
    for i in range(num):
        degree = (args.deg * i) % 360
        K, R = get_K_R(args.fov, degree, 0, resolution, resolution)
        Rs.append(R)
        Ks.append(K)

    K = torch.tensor(Ks).cuda()[None]
    R = torch.tensor(Rs).cuda()[None]

    images = torch.zeros((1, num, resolution, resolution, 3)).cuda()
    prompt = [scene_text] * num

    batch = {
        'images': images,
        'prompt': prompt,
        'R': R,
        'K': K
    }

    images_pred = model.inference(batch)
    
    # Create a directory for the scene inside the base directory
    scene_dir = os.path.join(base_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    # Save the prompt text to the scene folder
    with open(os.path.join(scene_dir, f'{scene_name}.txt'), 'w') as f:
        f.write(scene_text)

    # Save images for each perspective in the scene folder
    image_paths = []
    for i in range(num):
        im = Image.fromarray(images_pred[0, i])
        image_path = os.path.join(scene_dir, f'{i}.png')
        image_paths.append(image_path)
        im.save(image_path)

    # Generate video if the option is enabled
    generate_video(image_paths, scene_dir, args.fov, args.deg, args.gen_video, args.save_frames)

def main():
    args = parse_args()

    # Load the model
    config_file = 'configs/pano_generation.yaml'
    config = yaml.load(open(config_file, 'rb'), Loader=yaml.SafeLoader)
    model = PanoGenerator(config)
    model.load_state_dict(torch.load('weights/pano/augment.ckpt', map_location='cpu')['state_dict'], strict=False)
    model = model.cuda()

    resolution = config['dataset']['resolution']

    # Create the base results directory with a timestamp
    base_dir = os.path.join('outputs', f'results--{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(base_dir, exist_ok=True)

    # Iterate over each folder in the prompt directory
    prompt_folder = args.prompt_folder
    for scene_dir in os.listdir(prompt_folder):
        scene_path = os.path.join(prompt_folder, scene_dir)
        if os.path.isdir(scene_path):
            # Read the prompt text file (e.g., castle.txt)
            prompt_file = os.path.join(scene_path, f'{scene_dir}.txt')
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    scene_text = f.read().strip()
                print(f'Generating for scene: {scene_dir}')
                generate_for_scene(scene_dir, scene_text, args, model, resolution, config, base_dir)
            else:
                print(f'Skipping {scene_dir}, no prompt file found.')

if __name__ == '__main__':
    main()
