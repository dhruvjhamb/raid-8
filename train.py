"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import pathlib
import sys
import getpass
import argparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import time

from torch import nn

def parse():
    parser = argparse.ArgumentParser(description='Train or validate predefined models.')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--data', type=float, default=1.0)
    parser.add_argument('models', metavar='model', type=str, nargs='*')
    return parser.parse_args()

def validate(data_dir, data_transforms, num_classes,
    im_height, im_width, model=None):
    val_set = torchvision.datasets.ImageFolder(data_dir / 'val', data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1024, num_workers=4, pin_memory=True)

    if model == None:
        ckpt = torch.load('latest.pt')
        model = str_to_net[ckpt['model']](num_classes, im_height, im_width)
        model.load_state_dict(ckpt['net'])
        model.eval()
        print ("Number of parameters: {}, Time: {}, User: {}"
                        .format(ckpt['num_params'], ckpt['time'], ckpt['machine'])) 

    val_total, val_correct = 0, 0
    for idx, (inputs, targets) in enumerate(val_loader):
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        val_total += targets.size(0)
        val_correct += predicted.eq(targets).sum().item()
        print("\r", end='')
        print(f'validation {100 * idx / len(val_loader):.2f}%: {val_correct / val_total:.3f}', end='')

    return val_correct / val_total

def main():
    args = parse() 

    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    training_image_count = len(list(data_dir.glob('train/**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} total images'.format(image_count))
    print('Discovered {} training images'.format(training_image_count))
    print('Training with {} of the dataset ({} training images)'.format(
        args.data, int(args.data * training_image_count)))

    # Create the training data generator
    batch_size = 32
    num_batches = args.data * training_image_count / batch_size
    im_height = 64
    im_width = 64
    num_epochs = 1

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])

    if args.val:
        validate(data_dir, data_transforms, len(CLASS_NAMES),
            im_height, im_width)

    else:        
        assert len(args.models) <= 1, "If training, do not pass in more than one model."
        train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=4, pin_memory=True)

        if len(args.models) == 0 or str_to_net.get( args.models[0] ) == None:
            model_name = 'dummy'
        else:
            model_name = args.models[0]
        model = str_to_net[model_name](len(CLASS_NAMES), im_height, im_width)
        optim = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        for i in range(num_epochs):
            train_total, train_correct = 0,0
            for idx, (inputs, targets) in enumerate(train_loader):
                if idx > num_batches:
                    break
                optim.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optim.step()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                print("\r", end='')
                print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}', end='')
            ckpt_data = {
                'net': model.state_dict(),
                'model': model_name,
                'num_params': sum(p.numel() for p in model.parameters()),
                'time': time.time() - start_time,
                'num_epochs': i + 1,
                'machine': getpass.getuser(),
                'validation_acc': validate(data_dir, data_transforms,
                    len(CLASS_NAMES), im_height, im_width, model),
            }
            ckpt_file = 'latest.pt'.format()
            torch.save(ckpt_data, ckpt_file)


if __name__ == '__main__':
    main()