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
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def map_classes(dir_path):
    classes = os.listdir(dir_path)
    classes.sort()
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    return idx_to_class

def pairwise_equal(t):
    first_elem = torch.full(t.size(), t[0].item(), dtype=torch.long)
    return torch.equal(t, first_elem)

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

    data_transforms['flip'] = (transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        ]),
        0.01
    )

    class_map = map_classes(data_dir / 'train')
    transform_dir = pathlib.Path('./data/tiny-imagenet-transformed/train/')

    batch_size = IMAGES_PER_CLASS
    for transformation in data_transforms.keys():
        curr_transform_dir = transform_dir / transformation
        if args.overwrite:
            print ("Overwriting transformed images at {}".format(str(curr_transform_dir)))
            try_rmdir(curr_transform_dir)
        try_mkdir(curr_transform_dir)

        # Read in and transform all images
        data_transform, sampling_rate = data_transforms[transformation]
        train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=False, num_workers=1, pin_memory=True)

        curr_target = -1
        num_permutations = int(sampling_rate * batch_size)
        for idx, (inputs, targets) in enumerate(train_loader):
            assert(pairwise_equal(targets))
            target = targets[0].item()
            class_name = class_map[target]

            class_path = curr_transform_dir / class_name
            try_mkdir(class_path)
            images_path = class_path / 'images'
            try_mkdir(images_path)

            # Randomly sample images to permute
            indices = torch.randperm(batch_size)
            
            for index in indices[:num_permutations]:
                img_path = images_path / '{}_{}.JPEG'.format(class_name,
                    proper_order_int(str(index.item())))
                torchvision.utils.save_image(inputs[index.item()], img_path)

if __name__ == '__main__':
    main()
