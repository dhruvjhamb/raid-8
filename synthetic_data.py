"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import pathlib
import argparse
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import time

from torch import nn

from utils import *

IMAGES_PER_CLASS = 500

def parse():
    parser = argparse.ArgumentParser(description='Train or validate predefined models.')
#    parser.add_argument('--val', action='store_true')
#    parser.add_argument('--data', type=float, default=1.0)
#    parser.add_argument('--checkpoint', type=str, default='latest.pt')
#    parser.add_argument('models', metavar='model', type=str, nargs='*')
    return parser.parse_args()

def map_classes(dir_path):
    classes = os.listdir(dir_path)
    classes.sort()
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    return idx_to_class

def pairwise_equal(t):
    first_elem = torch.full(t.size(), t[0].item(), dtype=torch.long)
    return torch.equal(t, first_elem)

def try_mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def main():
    args = parse() 

    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    training_image_count = len(list(data_dir.glob('train/**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} total images'.format(image_count))
    print('Discovered {} training images'.format(training_image_count))

    # Create the training data generator
    im_height = 64
    im_width = 64

    # Key:      name of transformation
    # Value:    [tuple] (transform, frac. images to transform)
    #           e.g.    (transforms.Compose(...), 0.2)
    data_transforms = dict() 

    data_transforms['rotate'] = (transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        ]),
        0.2
    )

    class_map = map_classes(data_dir / 'train')
    transform_dir = pathlib.Path('./data/tiny-imagenet-transformed/train/')
    for transformation in data_transforms.keys():
        curr_transform_dir = transform_dir / transformation
        try_mkdir(curr_transform_dir)

        # Read in and transform all images
        data_transform, sampling_rate = data_transforms[transformation]
        train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=IMAGES_PER_CLASS,
                                                   shuffle=False, num_workers=1, pin_memory=True)

        curr_target = -1
        for idx, (inputs, targets) in enumerate(train_loader):
            assert(pairwise_equal(targets))
            target = targets[0].item()
            #if target != curr_target:
            #    curr_target = target
            #    class_counter = 0

            class_name = class_map[target]
            class_path = curr_transform_dir / class_name
            try_mkdir(class_path)
            images_path = class_path / 'images'
            try_mkdir(images_path)
            for index, image in enumerate(inputs):
                img_path = images_path / '{}_{}.JPEG'.format(class_name, proper_order_int(str(index)))
                torchvision.utils.save_image(image, img_path)

            #class_counter += 1

if __name__ == '__main__':
    main()
