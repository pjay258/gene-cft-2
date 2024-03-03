import sys
sys.path.append("..")

from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
import os
import glob
import upscaler
from gemini import generate_description
from utils import get_all_files_in_folder, conflict_samples
import google.generativeai as genai
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate upscaled images')
    parser.add_argument('-api_key', required=False, type=str, help="api key", default="AIzaSyAfkLYegbWFqCkbzxbvYQMvEotubfs4YdQ")
    parser.add_argument('-cuda', required=True, type=int, help="cuda number")
    parser.add_argument('-dataset', required=True, type=str, help="dataset")
    args = parser.parse_args()

    device = f'cuda:{args.cuda}'

    # Gemini API key 삽입
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel('gemini-pro-vision')

    # Upscale model 준비
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to(device)


    conflict_images_PATH = conflict_samples(args.dataset)
    cnt = 0
    for image_path in conflict_images_PATH:
        if args.dataset == 'bffhq':
            label = os.path.basename(image_path).split('_')[1]
            if label == '0': target_obj = 'young age'
            elif label == '1': target_obj = 'old age'
        elif args.dataset == 'cmnist':
            pass
        else: 
            print("===============label error================\n\n\n")
            breakpoint()

        generating_instruction_cnt = 0
        print(image_path)
        print(target_obj)
        description_error_flag = True
        upscaler_error_flag = True
        while True:
            try:
                description = generate_description(model=model, 
                                                   target_obj=target_obj,
                                                   image_PATH=image_path,
                                                   api_key=args.api_key)
                description_error_flag = False
                
                upscaler.upscaler(pipe=pipe,
                                  description=description,
                                  image_PATH=image_path)
                upscaler_error_flag = False
            except:
                generating_instruction_cnt += 1
                print("generating_cnt:", generating_instruction_cnt)
                print("description error:", description_error_flag)
                print("upscaler error:", upscaler_error_flag)
                description_error_flag = True
                upscaler_error_flag = True
                if generating_instruction_cnt > 100: break
                else: continue
            else:
                cnt += 1
                print((cnt*100)/len(conflict_images_PATH))
                print(description, '\n')
                break   

if __name__ == '__main__':
    main()
