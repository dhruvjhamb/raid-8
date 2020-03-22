import sys
import pathlib
from PIL import Image
import numpy as np
#import torch
#import torchvision.transforms as transforms
#from model import Net
import shutil
import os

def try_mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def proper_order(path):
    tokens = path.split('_')
    tokens[1] = tokens[1].zfill(3)
    return '_'.join(tokens)

def main():
    classes = dict()
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
#    ordered_dir = data_dir / 'train_ordered'
#    try_mkdir(ordered_dir)
    training_images = list(data_dir.glob('train/**/*.JPEG'))

    for idx, image in enumerate(training_images):
        print("\r", end='')
        print(f'Renamed {100 * idx / len(training_images):.2f}%', end='')
#        class_dir = ordered_dir / image.parts[-3]
#        try_mkdir(class_dir)
#        img_dir = class_dir / image.parts[-2]
#        try_mkdir(img_dir)
        ordered_img = image.parent / (proper_order(image.stem) + image.suffix)
        shutil.move(image, ordered_img)

if __name__ == '__main__':
	main()
