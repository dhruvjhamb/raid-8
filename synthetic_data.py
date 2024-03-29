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
from PIL import Image

from utils import *
from transform import *

TRAINING_IMAGES_PER_CLASS = 500
VAL_IMAGES_PER_CLASS = 50
NUM_CLASSES = 200

def _parse():
    parser = argparse.ArgumentParser(description='Train or validate predefined models.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save_transformations', action='store_true')
    parser.add_argument('--generate_samples', type=int, default=0)
    parser.add_argument('--transforms', metavar='transform',
            type=str, nargs='*')
    parser.add_argument('--data_type', type=str, default='train')
    return parser.parse_args()

def _validate_args(args):
    if not args.overwrite and args.generate_samples > 0:
        print ("Must overwrite if generating samples!")
        return False
    return True

def map_classes(dir_path):
    classes = os.listdir(dir_path)
    classes.sort()
    if '.DS_Store' in classes:
        classes.remove('.DS_Store')
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    return idx_to_class

def pairwise_equal(t):
    first_elem = torch.full(t.size(), t[0].item(), dtype=torch.long)
    return torch.equal(t, first_elem)

def get_generation_indices(num_batches, transform_batch_size, num_generated_samples):
    total_samples = num_batches * transform_batch_size
    generate_arr = np.zeros(total_samples)
    indices = np.random.choice(generate_arr.size, replace=False,
                           size=num_generated_samples)
    generate_arr[indices] = 1
    return generate_arr.reshape((num_batches, transform_batch_size))

def get_concat_h(im1_path, im2_path, gap=20):
    im1, im2 = Image.open(im1_path), Image.open(im2_path)
    dst = Image.new('RGB', (im1.width + im2.width + gap, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + gap, 0))
    return dst

def main():
    # Initialize arguments and timer
    args = _parse() 
    timer = Timer()

    if not _validate_args(args):
        return

    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    source_dir = data_dir / args.data_type

    image_count = len(list(data_dir.glob('**/*.JPEG')))
    training_image_count = len(list(data_dir.glob(args.data_type + '/**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in source_dir.glob('*')])
    print('Discovered {} total images'.format(image_count))
    print('Discovered {} training images'.format(training_image_count))

    # Create the training data generator
    im_height = 64
    im_width = 64

    class_map = map_classes(source_dir)
    transform_dir = pathlib.Path('./data/tiny-imagenet-transformed') / args.data_type

    if args.data_type == 'val':
        try_mkdir(transform_dir)

    num_batches = NUM_CLASSES
    shuffle = (not args.save_transformations)
    num_generated_samples = args.generate_samples
    if args.save_transformations: 
        if args.data_type == 'train':
            batch_size = TRAINING_IMAGES_PER_CLASS
        elif args.data_type == 'val':
            batch_size = VAL_IMAGES_PER_CLASS
        else:
            print('Not valid data type')
            return
    else:
        batch_size = num_generated_samples

    for transformation in args.transforms:
        if data_transforms.get(transformation) == None:
            continue

        data_transform, sampling_rate = data_transforms[transformation]

        if args.save_transformations:
            curr_transform_dir = transform_dir / transformation
        else:
            curr_transform_dir = transform_dir / "tmp"

        if args.overwrite:
            print ("Overwriting transformed images at {}".format(str(curr_transform_dir)))
            try_rmdir(curr_transform_dir)
        else:
            print ("Generating transformed images at {}".format(str(curr_transform_dir)))
        try_mkdir(curr_transform_dir)

        # If sampling rate is 0, do NOT do this transformation
        if sampling_rate == 0: continue


        train_set = torchvision.datasets.ImageFolder(source_dir, data_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=False, num_workers=1, pin_memory=True)

        curr_target = -1
        if args.save_transformations:
            num_permutations = int(sampling_rate * batch_size)
        else:
            num_permutations = num_generated_samples

        generation_indices = get_generation_indices(num_batches, num_permutations,
            num_generated_samples)        

        # Read in and transform all images
        for idx, (inputs, targets) in enumerate(train_loader):
            print("\r", end='')
            print(f'Generating transformed images from class {(idx + 1):.0f}', end='')
            
            if args.save_transformations: assert(pairwise_equal(targets))
            target = targets[0].item()
            class_name = class_map[target]

            class_path = curr_transform_dir / class_name
            try_mkdir(class_path)
            images_path = class_path / 'images'
            try_mkdir(images_path)

            source_path = source_dir / class_name / 'images'

            # Randomly sample images to permute
            indices = torch.randperm(batch_size)
            
            for gen_idx, index in enumerate(indices[:num_permutations]):
                img_name = '{}_{}.JPEG'.format(class_name,
                    proper_order_int(str(index.item())))
                img_path = images_path / img_name
                torchvision.utils.save_image(inputs[index.item()], img_path)

                if generation_indices[idx][gen_idx] == 1 or \
                        not args.save_transformations:    
                    # Concatenate source and transformed image, and display the result
                    get_concat_h(source_path / img_name, img_path).show()
            
            if not args.save_transformations: break

        

        if not args.save_transformations or (num_generated_samples > 0 and \
            not ask_yes_no("Do you want to keep the generated images")):
            print ("Removing images...")
            try_rmdir(curr_transform_dir)
            

if __name__ == '__main__':
    main()
