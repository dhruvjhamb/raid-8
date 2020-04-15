import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from utils import *
from transform import *

# Transforms file
# Key:      name of transformation
# Value:    [tuple] (transform, frac. images to transform)
#           e.g.    (transforms.Compose(...), 0.2)
data_transforms = dict() 

data_transforms['flip'] = (transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    ]),
    0.01
)

data_transforms['rotate10to15'] = (transforms.Compose([
    transforms.RandomRotation(degrees=(10,15), resample=Image.BICUBIC),
    transforms.ToTensor(),
    ]),
    0.01
)

data_transforms['rotate20to30'] = (transforms.Compose([
    transforms.RandomRotation(degrees=(20,30), resample=Image.BICUBIC),
    transforms.ToTensor(),
    ]),
    0.01
)

data_transforms['perspective'] = (transforms.Compose([
    transforms.RandomPerspective(p=1.0),
    transforms.ToTensor(),
    ]),
    0.01
)