import pathlib
import getpass
import argparse
import os
import math
from random import sample

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import time
import PIL
import matplotlib.pyplot as plt

from torch import nn

def _parse():
    parser = argparse.ArgumentParser(description='Plot model results from a checkpoint.')
    parser.add_argument('--checkpoint', metavar='checkpoint', type=str)
    return parser.parse_args()

def main():
    args = _parse() 
    ckpt_dir, cpt = './checkpoints/', args.checkpoint
    plots_dir = pathlib.Path('./plots')
    cptt = cpt.split('-')
    outfile = cptt[0].replace('/', '_')
    epoch = int(cptt[1][:-3])

    i, train_acc, val_acc, top5_acc, val_dx = 1, [], [0], [0], [0]
    while i <= epoch:
        ckpt = torch.load(ckpt_dir + cptt[0] + '-' + str(i) + '.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        train_acc.extend(ckpt['train_acc'])
        val_acc.append(ckpt['validation_acc'])
        top5_acc.append(ckpt['top5_validation'])
        val_dx.append(len(ckpt['train_acc']) * i - 1)
        i += 1
    
    fig, ax = plt.subplots()
    ax.plot(train_acc, label='train accuracy')
    ax.plot(val_dx, val_acc, label='validation accuracy')
    ax.plot(val_dx, top5_acc, label='top-5 accuracy')
    ax.legend()
    ax.set_xlabel('50-iterations', fontsize=16)
    ax.set_ylabel('accuracy', fontsize=16)
    plt.savefig(plots_dir / (outfile + '.png'))

if __name__ == '__main__':
    main()
