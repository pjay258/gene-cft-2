from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
import os
import glob

def upscaler(pipe, description: str, image_PATH: str):
    low_res_img = Image.open(image_PATH)
    low_res_img = low_res_img.resize((128, 128))

    upscaled_image = pipe(prompt=description, image=low_res_img).images[0]

    upscaled_save_PATH = image_PATH.replace('bffhq', 'upscaled-bffhq')
    upscaled_image.save(upscaled_save_PATH)


def rescale(folder_PATH, save_PATH):
    pattern = os.path.join(folder_PATH, '*.*')
    file_paths = [file for file in glob.glob(pattern)]

    cnt = 0
    for image_PATH in file_paths:
        cnt += 1
        image = Image.open(image_PATH)
        image = image.resize((512, 512), Image.BICUBIC)

        image_name = image_PATH.split('/')[-1].split('.')[0]
        image.save(f'{save_PATH}{image_name}.png')
        if cnt % 20 == 0: print((cnt*100)/len(file_paths))