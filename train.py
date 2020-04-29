"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import pathlib
import getpass
import argparse
import os
import math

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import time

from torch import nn

TRAINING_MOVING_AVG = 5

def parse():
    parser = argparse.ArgumentParser(description='Train or validate predefined models.')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--data', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default='latest.pt', nargs='+')
    parser.add_argument('models', metavar='model', type=str, nargs='*')
    parser.add_argument('--transforms', metavar='transform',
            type=str, nargs='*')
    return parser.parse_args()

def validate(data_dir, data_transforms, num_classes,
    im_height, im_width, checkpoint=None, model=None):

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_set = torchvision.datasets.ImageFolder(data_dir / 'val', data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)

    if model == None:
        temporary_models = []
        for cpt in checkpoint:
            ckpt_dir = './checkpoints/'
            ckpt = torch.load(ckpt_dir + cpt, map_location=device)
            model = str_to_net[ckpt['model']](num_classes, im_height, im_width)

            model.load_state_dict(ckpt['net'])
            model.eval()
            print ("Number of parameters: {}, Time: {}, User: {}"
                            .format(ckpt['num_params'], ckpt['runtime'], ckpt['machine'])) 
            temporary_models.append(model)
        model = temporary_models

    else:
        model = [model]

    for mod in model:
    # For GPU
        if device.type == 'cuda':
            mod.to(device)
        mod.eval()

    val_total, val_correct = 0, 0
    for idx, (inputs, targets) in enumerate(val_loader):
        all_predictions = None
        for mod in model:
            outputs = mod(inputs.to(device))
            _, predicted = outputs.max(1)
            if all_predictions is None:
                all_predictions = predicted.view(1, predicted.shape[0])
            else:
                all_predictions = torch.cat((all_predictions, predicted.view(1, predicted.shape[0])), 0)
        popular_vote = torch.mode(all_predictions, dim=0)[0]
        val_total += targets.size(0)
        val_correct += popular_vote.eq(targets.to(device)).sum().item()
        print("\r", end='')
        print(f'validation {100 * idx / len(val_loader):.2f}%: {val_correct / val_total:.3f}', end='')

    return val_correct / val_total

def main():

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    args = parse() 

    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    trans_data_dir = pathlib.Path('./data/tiny-imagenet-transformed')
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
    num_epochs = int(math.ceil(args.data))

    im_height = 224
    im_width = 224
    data_transforms = transforms.Compose([
        transforms.Resize((im_height, im_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), tuple(np.sqrt((0.229, 0.224, 0.255)))),
    ])

    if args.val:
        validate(data_dir, data_transforms, len(CLASS_NAMES),
            im_height, im_width, checkpoint=args.checkpoint)

    else:        
        assert len(args.models) <= 1, "If training, do not pass in more than one model."
        train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
        datasets = [train_set]
        if args.transforms != None:
            for transformation in args.transforms:
                try:
                    transform_dir = trans_data_dir / 'train' / transformation
                    print ("Reading transformed data from {}".format(transform_dir))
                    trans_set = torchvision.datasets.ImageFolder(transform_dir, data_transforms)
                    print ("Read {} transformed samples"
                            .format( len(trans_set) ))
                    datasets.append(trans_set)
                except:
                    print ("Reading transformed data FAILED, this data may not exist or may have a different name")

        complete_dataset = torch.utils.data.ConcatDataset(datasets)
        print('Discovered {} training samples (original and transformed)'
                .format( len(complete_dataset) ))
        train_loader = torch.utils.data.DataLoader(complete_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4, pin_memory=True)

        if len(args.models) == 0:
            model_name = 'dummy'
            print ("No model specified, defaulting to {}".format(model_name))
        elif str_to_net.get( args.models[0] ) == None:
            model_name = 'dummy'
            print ("Model {} does not exist, defaulting to {}".format(args.models[0],
                model_name))
        else:
            model_name = args.models[0]
        model = str_to_net[model_name](len(CLASS_NAMES), im_height, im_width)

        # For GPU
        if device.type == 'cuda':
            model.to(device)


        if model_name != 'dummy':   # changed from 'alex' to generalize
            params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params.append(param)
            optim = torch.optim.Adam(params)
        else:
            optim = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        model.train()
        start_time = time.time()
        for i in range(num_epochs):
            print ("Epoch {}...".format(i))
            train_total, train_correct = [],[]
            for idx, (inputs, targets) in enumerate(train_loader):
                if idx > num_batches:
                    break
                # gpu
                optim.zero_grad()
                outputs = model(inputs.to(device))
                loss = criterion(outputs.to(device), targets.to(device))
                loss.backward()
                optim.step()
                _, predicted = outputs.max(1)
                train_total.append (targets.size(0))
                train_correct.append (predicted.eq(targets.to(device)).sum().item())
                
                if (len(train_total) > TRAINING_MOVING_AVG):
                    train_total.pop(0)
                    train_correct.pop(0)

                moving_avg = sum(train_correct) / sum(train_total)

                print("\r", end='')
                print(f'training {100 * idx / len(train_loader):.2f}%: {100 * moving_avg:.2f}', end='')

            ckpt_data = {
                'net': model.state_dict(),
                'model': model_name,
                'num_params': sum(p.numel() for p in model.parameters()),
                'runtime': time.time() - start_time,
                'timestamp': start_time,
                'epoch': i + 1,
                'machine': getpass.getuser(),
                'validation_acc': validate(data_dir, data_transforms,
                    len(CLASS_NAMES), im_height, im_width, model=model),
            }
            
            checkpoint_dir = pathlib.Path('./checkpoints') / ckpt_data['model']
            if not os.path.isdir(pathlib.Path('./checkpoints')):
                os.mkdir(pathlib.Path('./checkpoints'))
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            ckpt_file = '{}-{}.pt'.format(
                ckpt_data['timestamp'], ckpt_data['epoch'])
            torch.save(ckpt_data, checkpoint_dir / ckpt_file)

            print ()

if __name__ == '__main__':
    main()
