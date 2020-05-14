import pathlib
import getpass
import argparse
import os
import math
from random import sample

import PIL
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser(description='Plot bar graph from a list of values.')
    parser.add_argument('--name', metavar='name', type=str)
    parser.add_argument('--values', metavar='values', type=float, nargs='*')
    parser.add_argument('--labels', metavar='labels', type=str, nargs='*')
    parser.add_argument('--x_axis', metavar='x_axis', type=str)
    parser.add_argument('--y_axis', metavar='y_axis', type=str)
    return parser.parse_args()

def main():
    args = parse()
    name, labels, values, x_axis, y_axis = args.name, args.labels, args.values, args.x_axis, args.y_axis
    if len(values) != len(labels):
        print("arguments don't match")
        return
    if '.png' not in name:
        print("need .png in --name")
        return
    fig, ax = plt.subplots()
    ax.set(ylim=[85, 95])
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    ax.bar(labels, values)
    ax.set_xlabel(x_axis, fontsize=16)
    ax.set_ylabel(y_axis, fontsize=16)
    plots_dir = pathlib.Path('./plots')
    plt.savefig(plots_dir / name)

if __name__ == '__main__':
    main()
