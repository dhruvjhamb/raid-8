import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from utils import *
from transform import *

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Transforms file
# Key:      name of transformation
# Value:    [tuple] (transform, frac. images to transform)
#           e.g.    (transforms.Compose(...), 0.2)
data_transforms = dict() 

data_transforms['flip'] = (transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    ]),
    1
)

data_transforms['jitter'] = (transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor(),
    ]),
    1
)

data_transforms['rotate10to15'] = (transforms.Compose([
    transforms.Pad(32, padding_mode='reflect'),
    transforms.RandomRotation(degrees=(10,20), resample=Image.BICUBIC),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    ]),
    0.01
)

data_transforms['rotate20to30'] = (transforms.Compose([
    transforms.Pad(32, padding_mode='reflect'),
    transforms.RandomRotation(degrees=(20,30), resample=Image.BICUBIC),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    ]),
    1
)

data_transforms['rotate30to45'] = (transforms.Compose([
    transforms.Pad(32, padding_mode='reflect'),
    transforms.RandomRotation(degrees=(30,45), resample=Image.BICUBIC),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    ]),
    1
)

data_transforms['flip-gaussian'] = (transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    GaussianNoise(0., 0.05),
    ]),
    1
)

data_transforms['jitter-gaussian'] = (transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor(),
    GaussianNoise(0., 0.05),
    ]),
    1
)

data_transforms['rotate10to15-gaussian'] = (transforms.Compose([
    transforms.Pad(32, padding_mode='reflect'),
    transforms.RandomRotation(degrees=(10,20), resample=Image.BICUBIC),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    GaussianNoise(0., 0.05),
    ]),
    0.01
)

data_transforms['rotate20to30-gaussian'] = (transforms.Compose([
    transforms.Pad(32, padding_mode='reflect'),
    transforms.RandomRotation(degrees=(20,30), resample=Image.BICUBIC),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    GaussianNoise(0., 0.05),
    ]),
    1
)
data_transforms['rotate30to45-gaussian'] = (transforms.Compose([
    transforms.Pad(32, padding_mode='reflect'),
    transforms.RandomRotation(degrees=(30,45), resample=Image.BICUBIC),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    GaussianNoise(0., 0.05),
    ]),
    1
)

data_transforms['perspective'] = (transforms.Compose([
    # transforms.Pad(32, padding_mode='reflect'),
    transforms.RandomPerspective(p=1.0),
    # transforms.CenterCrop(64),
    transforms.ToTensor(),
    ]),
    1
)

data_transforms['gaussian'] = (transforms.Compose([
    transforms.ToTensor(),
    GaussianNoise(0., 0.05),
    # transforms.ToTensor(),
    ]),
    1
)
