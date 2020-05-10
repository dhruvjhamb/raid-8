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
import sys
import math
from random import sample

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import time
import PIL

from torch import nn

TRAINING_MOVING_AVG = 5
MOVING_AVG_LENGTH = 2
OVERFIT_THRES = 0.95

def parse():
    parser = argparse.ArgumentParser(description='Train or validate predefined models.')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--cross_val', action='store_true')
    parser.add_argument('--data', type=float, default=1.0)
    parser.add_argument('--checkpoints', type=str, nargs='+')
    parser.add_argument('--checkpoint_weights', type=float, nargs='+')
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
    parser.add_argument('--decaythres', type=float)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--true_epoch', action='store_true')
    parser.add_argument('--batchnorm_lr', type=float)
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
    if interpolation is None or \
            mapping.get(interpolation) == None:
        print ("Using default interpolation (bicubic)")
        result = mapping["bicubic"]
    else:
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

def getCheckpointFileName(ckpt_data):
    checkpoint_dir = pathlib.Path('./checkpoints') / ckpt_data['model']
    try_mkdir(pathlib.Path('./checkpoints'))
    try_mkdir(checkpoint_dir)
    ckpt_file = '{}-{}.pt'.format(
        ckpt_data['timestamp'], ckpt_data['epoch'])
    return checkpoint_dir / ckpt_file

def saveCheckpoint(ckpt_data):
    filename = getCheckpointFileName(ckpt_data)
    torch.save(ckpt_data, filename)

def initializeLogging(logfile, model_name):
    if logfile is not None:
        log_dir = pathlib.Path('./logs')
        try_mkdir(log_dir)
        file_path = log_dir / (logfile + ".txt")

        f = open(file_path, "w+")
        f.write("Model Name: {} \n".format(model_name))
    else:
        f = None
    return f

def logCheckpoint(f, ckpt_data):
    if f == None: return
    # write metrics to text file if logfile arg not None
    for key in ckpt_data.keys():
        if key != "net" and key != "train_acc":
            output = key + " {}\n"
            f.write(output.format(ckpt_data[key]))
    f.write("\n")

def isModelOverfitting(history):
    if len(history) <= MOVING_AVG_LENGTH + 1:
        return False
    avgs = []
    for i in range(len(history) - MOVING_AVG_LENGTH + 1):
        window = history[i:i+MOVING_AVG_LENGTH]
        avgs.append(
            sum(window) / len(window)
            )
    return any([(avgs[-1] < OVERFIT_THRES * window) for window in avgs])

# def sampleMode(predictions):
#     samples = []
#     for model_pred in predictions:
#         modes = multimode(model_pred)
#         samples.append(sample(modes, 1)[0])
#     return samples

# def k_accuracy(outputs, targets, k):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     batch_size = targets.to(device).size(0)
#     _, pred = outputs.to(device).topk(k, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(targets.to(device).view(1, -1).expand_as(pred.to(device)))
#     correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#     return correct_k.mul_(100.0 / batch_size).item() / 100.0

def validate(data_dir, data_transforms, num_classes,
    im_height, im_width, args_transforms=None, checkpoint=None, model=None, random_choice=False, weights=None):
    
    load_from_ckpt = (model == None)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if load_from_ckpt: batch_size = 4
    else: batch_size = 32

    if args_transforms != None:
        trans_data_dir = pathlib.Path('./data/tiny-imagenet-transformed')
        image_count = len(list(data_dir.glob('**/*.JPEG')))
        val_image_count = len(list(data_dir.glob('val/**/*.JPEG')))
        CLASS_NAMES = np.array([item.name for item in (data_dir / 'val').glob('*')])

        val_set = torchvision.datasets.ImageFolder(data_dir / 'val', data_transforms)
        datasets = [val_set]
        for transformation in args_transforms:
            try:
                transform_dir = trans_data_dir / 'val' / transformation
                print ("Reading transformed data from {}".format(transform_dir))
                trans_set = torchvision.datasets.ImageFolder(transform_dir, data_transforms)
                print ("Read {} transformed samples"
                        .format( len(trans_set) ))
                datasets.append(trans_set)
            except:
                print ("Reading transformed data FAILED, this data may not exist or may have a different name")
        if weights != None:
            datasets = reweightDatasets(datasets, [1] + weights)

        val_set = torch.utils.data.ConcatDataset(datasets)
    else:
        val_set = torchvision.datasets.ImageFolder(data_dir / 'val', data_transforms)
    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)

    if load_from_ckpt:
        temporary_models = []
        for cpt in checkpoint:
            ckpt_dir = './checkpoints/'
            ckpt = torch.load(ckpt_dir + cpt, map_location=device)
            model = str_to_net[ckpt['model']](num_classes, im_height, im_width, None)

            model.load_state_dict(ckpt['net'])
            print ("Number of parameters: {}, Time: {}, User: {}"
                            .format(ckpt['num_params'], ckpt['runtime'], ckpt['machine'])) 
            temporary_models.append(model)
            del ckpt
        model = temporary_models

    else:
        model = [model]

    if weights == None:
        weights = np.ones(len(model))/len(model)

    assert len(model) == len(weights), "There must be one weight for each model."
    # assert np.isclose(np.sum(weights), 1), "Weights must sum to 1."

    for mod in model:
    # For GPU
        if device.type == 'cuda':
            mod.to(device)
        mod.eval()

    torch.cuda.empty_cache()
    val_total, val_correct, val_topk_correct = 0, 0, 0
    k=5
    for idx, (inputs, targets) in enumerate(val_loader):
        sum_probabilities = None
        for i in range(len(model)):
            outputs = model[i](inputs.to(device))
            probabilites = (nn.Softmax(dim=-1)(outputs)).to(device)
            weighted_prob = (probabilites*weights[i]).to(device)
            if sum_probabilities is None:
                sum_probabilities = weighted_prob
            else:
                sum_probabilities = sum_probabilities + weighted_prob
                sum_probabilities = sum_probabilities.to(device)

        _, predicted = sum_probabilities.max(1)
        _, predicted_topk = sum_probabilities.topk(k, 1, True, True)
        predicted_topk = predicted_topk.t()
        correct = predicted_topk.eq(targets.to(device).view(1, -1).expand_as(predicted_topk.to(device)))
        val_topk_correct += correct[:k].view(-1).float().sum(0).item()
        val_correct += predicted.eq(targets.to(device)).sum().item()
        val_total += targets.size(0)

        print("\r", end='')
        print(f'validation {100 * idx / len(val_loader):.2f}% complete top-1: {val_correct / val_total:.3f} top-5: {val_topk_correct / val_total:.3f}', end='')

    return val_correct / val_total, val_topk_correct / val_total


def cross_validate(data_dir, data_transforms, num_classes, im_height, im_width, args):
    if args.transforms == None:
        print('Need to specify transforms')
        return
    for i in range(len(args.transforms)):
        print('Training with:', args.transforms[:i] + args.transforms[i+1:])
        model = train(data_dir, data_transforms, args.transforms[:i] + args.transforms[i+1:], num_classes, im_height, im_width, args, val=False)
        print('Validation with:', args.transforms[i])
        validate(data_dir, data_transforms, num_classes, im_height, im_width, args_transforms=[args.transforms[i]], model=model)


def train(data_dir, data_transforms, args_transforms, num_classes,
    im_height, im_width, args, model=None, random_choice=False, val=True):

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans_data_dir = pathlib.Path('./data/tiny-imagenet-transformed')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    training_image_count = len(list(data_dir.glob('train/**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])

    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
    datasets = [train_set]
    if args_transforms != None:
        for transformation in args_transforms:
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
    complete_data_len = len(complete_dataset)
    print('Discovered {} training samples (original and transformed)'
            .format( complete_data_len ))

    # Create the training data generator
    batch_size = args.batchsize
    train_loader = torch.utils.data.DataLoader(complete_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    # If args.data > 1, num_batches will have no impact on training
    if args.true_epoch:
        num_batches = args.data * complete_data_len / batch_size
    else:
        num_batches = args.data * training_image_count / batch_size
    num_epochs = int(math.ceil(args.data))

    params = {'lrs': args.learningrates, 
            'partitions': args.partitions, 
            'bn_lr': args.batchnorm_lr,
            'decay_schedule': {
                'decay_rate': args.decayrate,
                'decay_coeff': args.decaycoeff,
                'decay_thres': args.decaythres
                }
            }

    model_args = (len(CLASS_NAMES), im_height, im_width, params)
    if len(args.checkpoints) == 1:
        model, model_name = getModelFromCheckpoint(args.checkpoints[0], model_args, device)
    else:
        model, model_name = getModelFromName(args.models, model_args)

    # For GPU
    if device.type == 'cuda':
        model.to(device)

    optim = torch.optim.Adam(model.optim_params)
    criterion = nn.CrossEntropyLoss()
    model.train()
    start_time = time.time()

    f = initializeLogging(args.logfile, model_name)

    val_history = []
    for i in range(num_epochs):
        print ("Epoch {}...".format(i))
        train_total, train_correct, train_acc = [], [], []
        for idx, (inputs, targets) in enumerate(train_loader):
            if idx > num_batches:
                break
            
            # copy to gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            # gpu
            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            _, predicted = outputs.max(1)

            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()
            train_total.append (total)
            train_correct.append (correct)

            if (len(train_total) > TRAINING_MOVING_AVG):
                train_total.pop(0)
                train_correct.pop(0)

            moving_avg = sum(train_correct) / sum(train_total)

            print("\r", end='')
            print(f'[{100 * idx / len(train_loader):.2f}%] acc: {100 * moving_avg:.2f}, loss: {loss:.2f}', end='')
            train_acc.append(100. * correct / total)

        if val:
            val_acc, top5_acc = validate(data_dir, data_transforms, len(CLASS_NAMES), im_height, im_width, model=model)
            val_history.append(val_acc)
            optim = decayLR(optim, i, model, val_history)

            ckpt_data = {
                'net': model.state_dict(),
                'command': ' '.join(sys.argv),
                'model': model_name,
                'checkpoint_file': '', 
                'num_params': sum(p.numel() for p in model.parameters()),
                'runtime': time.time() - start_time,
                'timestamp': start_time,
                'epoch': i + 1,
                'machine': getpass.getuser(),
                'train_acc': train_acc,
                'validation_acc': val_acc * 100.,
                'top5_validation': top5_acc * 100.,
                'model_args': vars(args),
            }

        else:
            ckpt_data = {
                'net': model.state_dict(),
                'command': ' '.join(sys.argv),
                'model': model_name,
                'checkpoint_file': '', 
                'num_params': sum(p.numel() for p in model.parameters()),
                'runtime': time.time() - start_time,
                'timestamp': start_time,
                'epoch': i + 1,
                'machine': getpass.getuser(),
                'train_acc': train_acc,
                'model_args': vars(args),
            }

        filename = str(getCheckpointFileName(ckpt_data))
        ckpt_data['checkpoint_file'] = filename
        
        saveCheckpoint(ckpt_data)
        logCheckpoint(f, ckpt_data)
        print ()
        if isModelOverfitting(val_history):
            break

    if f is not None:
        f.close()

    return model


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


    im_height, im_width = 224, 224

    interpolation = getInterpolationMethod(args.interpolate)
    data_transforms = transforms.Compose([
        transforms.Resize((im_height, im_width), interpolation=interpolation),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), tuple(np.sqrt((0.229, 0.224, 0.255)))),
    ])

    if args.val:
        validate(data_dir, data_transforms, len(CLASS_NAMES),
            im_height, im_width, checkpoint=args.checkpoints, weights=args.checkpoint_weights)
        
    elif args.cross_val:
        assert len(args.models) <= 1, "If training, do not pass in more than one model."
        if args.checkpoints != None:
            assert args.checkpoints == None or len(args.checkpoints) <= 1, "If training, do not pass in more than one checkpoint."
            assert len(args.models) + len(args.checkpoints) <= 1, "Cannot pass in both a model and " \
                + "a checkpoint."
        else:
            args.checkpoints = []
            
        cross_validate(data_dir, data_transforms, len(CLASS_NAMES), im_height, im_width, args)

    else:        
        assert len(args.models) <= 1, "If training, do not pass in more than one model."
        if args.checkpoints != None:
            assert args.checkpoints == None or len(args.checkpoints) <= 1, "If training, do not pass in more than one checkpoint."
            assert len(args.models) + len(args.checkpoints) <= 1, "Cannot pass in both a model and " \
                + "a checkpoint."
        else:
            args.checkpoints = []

        train(data_dir, data_transforms, args.transforms, len(CLASS_NAMES),
            im_height, im_width, args)

if __name__ == '__main__':
    main()
