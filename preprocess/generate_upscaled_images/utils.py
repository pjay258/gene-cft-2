import glob
import os
from PIL import Image
import pandas as pd

def parse_instructions(instructions_string):
    instructions_dict = {}

    for instruction in instructions_string.split('\n'):
        if not instruction.strip():
            continue

        key, value = map(str.strip, instruction.split(':', 1))

        key = key.lstrip('* ').strip()

        instructions_dict[key] = value

    return instructions_dict


def get_all_files_in_folder(folder_path):
    pattern = os.path.join(folder_path, '*.*')
    file_paths = [file for file in glob.glob(pattern)]

    return file_paths


def unprocessed_images(folder_PATH, save_PATH):
    reprocess_files = get_all_files_in_folder(folder_PATH)
    postprocess_files = get_all_files_in_folder(save_PATH)

    reprocess_files_name = [os.path.basename(file) for file in reprocess_files]
    postprocess_files_name = [os.path.basename(file) for file in postprocess_files]

    unprocessed_files = []
    for file in reprocess_files_name:
        if file not in postprocess_files_name:
            unprocessed_files.append(f'{folder_PATH}/{file}')

    return unprocessed_files


def conflict_samples(dataset):
    df = pd.read_csv(f'../PBA/bias_tables/{dataset}/bias_table.csv')
    conflict_files = df[df['bias'] == 0]['file_path'].tolist()
    
    return conflict_files
