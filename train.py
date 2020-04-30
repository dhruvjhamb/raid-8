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
import PIL

from torch import nn

TRAINING_MOVING_AVG = 5

def parse():
    parser = argparse.ArgumentParser(description='Train or validate predefined models.')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--data', type=float, default=1.0)
    parser.add_argument('--checkpoints', type=str, nargs='+')
    parser.add_argument('--interpolate', type=str, nargs="?")
    parser.add_argument('models', metavar='model', type=str, nargs='*')
    parser.add_argument('--transforms', metavar='transform',
            type=str, nargs='*')
    parser.add_argument('--weights', metavar='transweights',
            type=float, nargs='*')
    parser.add_argument('--logfile', metavar=str, nargs="?")
    parser.add_argument('--learningrates', metavar='lr',
            type=float, nargs='*')
    parser.add_argument('--partitions', metavar='partition',
            type=float, nargs='*')
    parser.add_argument('--decayrate', type=int, default=1)
    parser.add_argument('--decaycoeff', type=float, default=1.0)
    parser.add_argument('--batchsize', type=int, default=32)
    return parser.parse_args()

def reweightDatasets(datasets, weights):
    reweighted = []
    source_samples = len(datasets[0])
    for index, dataset in enumerate(datasets):
        target_samples = int(source_samples * weights[index])
        curr_samples = len(dataset)
        indices = np.random.permutation(curr_samples)[:target_samples]

        reweighted.append(
                torch.utils.data.Subset(dataset, indices)
                )
    return reweighted

def getInterpolationMethod(interpolation):
    mapping = {"hamming": PIL.Image.HAMMING,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "bilinear": PIL.Image.BILINEAR}
    if mapping.get(interpolation) == None:
        print ("Using default interpolation (bicubic)")
        result = mapping["bicubic"]
    print ("Using {} interpolation".format(interpolation))
    result = mapping[interpolation]
    return result

def getModelFromCheckpoint(cpt, args, device):
    ckpt_dir = './checkpoints/'
    ckpt = torch.load(ckpt_dir + cpt, map_location=device)
    model = str_to_net[ckpt['model']](*args)

    model.load_state_dict(ckpt['net'])
    return model, ckpt['model']

def getModelFromName(model, args):
    if len(model) == 0:
        model_name = 'dummy'
        print ("No model specified, defaulting to {}".format(model_name))
    elif str_to_net.get( model[0] ) == None:
        model_name = 'dummy'
        print ("Model {} does not exist, defaulting to {}".format(model[0],
            model_name))
    else:
        model_name = model[0]
    return str_to_net[model_name](*args), model_name

def validate(data_dir, data_transforms, num_classes,
    im_height, im_width, checkpoint=None, model=None):
    
    load_from_ckpt = (model == None)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if load_from_ckpt: batch_size = 16
    else: batch_size = 128
    val_set = torchvision.datasets.ImageFolder(data_dir / 'val', data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)

    if load_from_ckpt:
        temporary_models = []
        for cpt in checkpoint:
            ckpt_dir = './checkpoints/'
            ckpt = torch.load(ckpt_dir + cpt)
            model = str_to_net[ckpt['model']](num_classes, im_height, im_width, None)

            model.load_state_dict(ckpt['net'])
            print ("Number of parameters: {}, Time: {}, User: {}"
                            .format(ckpt['num_params'], ckpt['runtime'], ckpt['machine'])) 
            temporary_models.append(model)
            del ckpt
        model = temporary_models

    else:
        model = [model]

    for mod in model:
    # For GPU
        if device.type == 'cuda':
            mod.to(device)
        mod.eval()

    torch.cuda.empty_cache()
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
    batch_size = args.batchsize
    num_batches = args.data * training_image_count / batch_size
    im_height = 64
    im_width = 64
    num_epochs = int(math.ceil(args.data))

    im_height = 224
    im_width = 224

    if args.interpolate is not None:
        interpolation = getInterpolationMethod(args.interpolate)
        data_transforms = transforms.Compose([
            transforms.Resize((im_height, im_width), interpolation=interpolation),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), tuple(np.sqrt((0.229, 0.224, 0.255)))),
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((im_height, im_width)),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), tuple(np.sqrt((0.229, 0.224, 0.255)))),
        ])

    if args.val:
        validate(data_dir, data_transforms, len(CLASS_NAMES),
            im_height, im_width, checkpoint=args.checkpoints)

    else:        
        assert len(args.models) <= 1, "If training, do not pass in more than one model."
        if args.checkpoints != None:
            assert args.checkpoints == None or len(args.checkpoints) <= 1, "If training, do not pass in more than one checkpoint."
            assert len(args.models) + len(args.checkpoints) <= 1, "Cannot pass in both a model and " \
                + "a checkpoint."
        else:
            args.checkpoints = []

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
        if args.weights != None:
            datasets = reweightDatasets(datasets, [1] + args.weights)

        complete_dataset = torch.utils.data.ConcatDataset(datasets)
        print('Discovered {} training samples (original and transformed)'
                .format( len(complete_dataset) ))
        train_loader = torch.utils.data.DataLoader(complete_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4, pin_memory=True)

        params = {'lrs': args.learningrates, 'partitions': args.partitions, 'decay_schedule': 
                {'decay_rate': args.decayrate, 'decay_coeff': args.decaycoeff}}
        model_args = (len(CLASS_NAMES), im_height, im_width, params)
        if len(args.checkpoints) == 1:
            model, model_name = getModelFromCheckpoint(args.checkpoints[0], model_args, device)
        else:
            model, model_name = getModelFromName(args.models, model_args)

        # For GPU
        if device.type == 'cuda':
            model.to(device)

        if model_name != 'dummy':   # changed from 'alex' to generalize
            optim = torch.optim.Adam(model.optim_params)
        else:
            optim = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        model.train()
        start_time = time.time()

        if args.logfile is not None:
            if not os.path.exists('./logs/'):
                os.makedirs('./logs/')

            f = open("./logs/" + args.logfile + ".txt", "w+")
            f.write("Model Name: {} \n".format(model_name))

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

            val_acc = validate(data_dir, data_transforms, len(CLASS_NAMES), im_height, im_width, model=model)

            optim = decayLR(optim, i, model)

            ckpt_data = {
                'net': model.state_dict(),
                'model': model_name,
                'num_params': sum(p.numel() for p in model.parameters()),
                'runtime': time.time() - start_time,
                'timestamp': start_time,
                'epoch': i + 1,
                'machine': getpass.getuser(),
                'validation_acc': val_acc,
            }
            
            checkpoint_dir = pathlib.Path('./checkpoints') / ckpt_data['model']
            if not os.path.isdir(pathlib.Path('./checkpoints')):
                os.mkdir(pathlib.Path('./checkpoints'))
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            ckpt_file = '{}-{}.pt'.format(
                ckpt_data['timestamp'], ckpt_data['epoch'])
            torch.save(ckpt_data, checkpoint_dir / ckpt_file)

            # write metrics to text file if logfile arg not None
            if args.logfile is not None:
                f.write("Epoch {} \n".format(ckpt_data['epoch']))
                f.write("Validation Accuracy: {} \n".format(ckpt_data['validation_acc']))

            print ()

        if args.logfile is not None:
            f.close()

if __name__ == '__main__':
    main()
