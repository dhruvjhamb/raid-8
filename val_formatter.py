import sys
import pathlib
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from model import Net
import shutil
import os

def main():
	classes = dict()
	data_dir = pathlib.Path('./data/tiny-imagenet-200/val/')
	img_dir = pathlib.Path('./data/tiny-imagenet-200/val/images/')
	with open(data_dir / 'val_annotations.txt', 'r') as fp:
		line = fp.readline()
		cnt = 1
		while line:
			if cnt % 50 == 0:
				print("Line {} processed".format(cnt))
			line = fp.readline()
			tokens = line.split()
			if len(tokens) < 6:
				break
			if classes.get(tokens[1]) == None:
				classes[tokens[1]] = data_dir / tokens[1]
				os.mkdir(classes[tokens[1]])
			shutil.copy(img_dir / tokens[0], classes[tokens[1]])
			cnt += 1

if __name__ == '__main__':
	main()