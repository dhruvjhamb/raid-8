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
    parser.add_argument('--checkpoint', metavar='checkpoint', type=str, nargs='*')
    parser.add_argument('--top5', action='store_true')
    return parser.parse_args()

def main():
    args = _parse() 
    for i, cpt in enumerate(args.checkpoint):
        ckpt_dir = './checkpoints/'
        plots_dir = pathlib.Path('./plots')
        cptt = cpt.split('-')
        outfile = cptt[0].replace('/', '_')
        epoch = int(cptt[1][:-3])

        i, train_acc, train_loss, val_acc, top5_acc, val_dx = 1, [], [], [None], [None], [0]
        while i <= epoch:
            ckpt = torch.load(ckpt_dir + cptt[0] + '-' + str(i) + '.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            if ckpt.get('train_acc') is not None:
                train_acc.extend(ckpt['moving_train_acc'])
                train_loss.extend(ckpt['train_loss'])
                val_dx.append(len(ckpt['train_acc']) * i - 1)
            val_acc.append(ckpt['validation_acc'])
            if ckpt.get('top5_validation') is not None:
                top5_acc.append(ckpt['top5_validation'])
            i += 1
    
        fig, ax = plt.subplots()
        # Only plot train and top5 accuracies if all checkpoints had data for them
        if len(train_acc) > 0:
            ax.plot(train_acc, label='moving train accuracy')
        if len(val_dx) == len(val_acc):
            ax.plot(val_dx, val_acc, label='validation accuracy', linewidth=8)
        else:
            ax.plot(val_acc, label='validation accuracy', linewidth=8)
        if len(top5_acc) == len(val_acc):
            ax.plot(val_dx, top5_acc, label='top-5 accuracy', linewidth=8)
        ax.legend()
        ax.set_xlabel('iterations', fontsize=16)
        ax.set_ylabel('accuracy', fontsize=16)
        plt.savefig(plots_dir / (outfile + '-acc.png'))

        fig1, ax1 = plt.subplots()
        if len(train_loss) > 0:
            ax1.plot(train_loss, label='train loss')
        ax.legend()
        ax.set_xlabel('iterations', fontsize=16)
        ax.set_ylabel('loss', fontsize=16)

        plt.savefig(plots_dir / (outfile + '-loss.png'))
     
    if len(args.checkpoint) > 1:
        fig, ax = plt.subplots()
        outfile = ''
        for cpt in args.checkpoint:
            cptt = cpt.split('-')
            outfile += cptt[0].replace('/', '_') + '_'
            epoch = int(cptt[1][:-3])
            i, val_acc, val_dx = 1, [None], [0]
            while i <= epoch:
                ckpt = torch.load(ckpt_dir + cptt[0] + '-' + str(i) + '.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                val_dx.append(i + 1)
                if not args.top5:
                    val_acc.append(ckpt['validation_acc'])
                else:
                    val_acc.append(ckpt['top5_validation'])
                i += 1
            if len(val_dx) == len(val_acc):
                ax.plot(val_dx, val_acc, label=cptt[0].replace('/', '_'))
            else:
                ax.plot(val_acc, label=cptt[0].replace('/', '_'))
        ax.legend()
        ax.set_xlabel('iterations', fontsize=16)
        ax.set_ylabel('accuracy', fontsize=16)
        plt.savefig(plots_dir / (outfile + '-multi.png'))            

if __name__ == '__main__':
    main()
