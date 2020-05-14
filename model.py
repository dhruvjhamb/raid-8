import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from utils import *
from statistics import pstdev
from collections import OrderedDict

WINDOW_LENGTH = 3

class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width, params=None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#       self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.layer1 = nn.Linear(im_height * im_width * 32, num_classes)
#       self.layer2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
#       x = self.conv2(x)
        x = x.flatten(1)
#       x = F.relu(self.layer1(x))
        x = self.layer1(x)
        return x

class DummyNet(nn.Module):
    def __init__(self, num_classes, im_height, im_width, params=None):
        super(DummyNet, self).__init__()
        self.layer1 = nn.Linear(im_height * im_width * 3, num_classes)
        self.optim_params = self.parameters()
        self.decay_schedule = params['decay_schedule']

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        return x

def AlexFineTuned(num_classes, im_height, im_width, params=None):
    alexnet = torchvision.models.alexnet(pretrained=True)
    for param in alexnet.parameters():
        param.requires_grad = False
    num_features = alexnet.classifier[6].in_features
    alexnet.classifier[6] = nn.Linear(num_features, num_classes)
    return alexnet

def ResNetFineTuned(num_classes, im_height, im_width, params=None):
    # finetuning - https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4
    resnet = torchvision.models.resnet18(pretrained=True)
    return ResNetCommon(resnet, num_classes, params)

def ResNet34FineTuned(num_classes, im_height, im_width, params=None):
    resnet = torchvision.models.resnet34(pretrained=True)
    return ResNetCommon(resnet, num_classes, params)

def ResNextFineTuned(num_classes, im_height, im_width, params=None):
    resnext50_32x4d = torchvision.models.resnext50_32x4d(pretrained=True)
    return ResNetCommon(resnext50_32x4d, num_classes, params)

def ResNetCommon(resnet, num_classes, params=None):
    ct = 0
    resnet.optim_params = []

    if params != None:
        # Initialize learning rates
        resnet.decay_schedule = params['decay_schedule']
        lrs = params['lrs']
        partitions = params['partitions']
        dropout_rate = params.get('dropout_rate')
        dropout_layers = params.get('dropout_size')

        freeze_all = (partitions is None)

        if not freeze_all:
            partition_assignment = partitionList(
                sum([1 for _ in resnet.named_children()]), partitions)

        # Set defaults
        if params.get('bn_lr') is None: bn_lr = 0
        else: bn_lr = params['bn_lr']

        # Save other parameters
        
        if not freeze_all:
            num_features = resnet.fc.in_features
            if dropout_rate is None:
                resnet.fc = nn.Linear(num_features, num_classes)
            else:
                dropout_layers = [num_features * i 
                        for i in dropout_layers]
                layers = [num_features] + dropout_layers + \
                        [num_classes]
                resnet.fc = createDropoutUnit(dropout_rate,
                        layers)

        for name, child in resnet.named_children():
            if not freeze_all:
                partition = partition_assignment[ct]
            for param_name, params in child.named_parameters():
                if freeze_all or lrs[partition] == 0:
                    if "bn" in param_name and \
                            bn_lr > 0:
                        optim_params = {'params': params}
                        optim_params['lr'] = bn_lr
                        resnet.optim_params.append(optim_params)
                    else:
                        params.requires_grad = False
                else:
                    optim_params = {'params': params}
                    optim_params['lr'] = lrs[partition]
                    resnet.optim_params.append(optim_params)
            ct += 1
        if freeze_all:
            num_features = resnet.fc.in_features
            resnet.fc = nn.Linear(num_features, num_classes)
    else:
        # Validation case
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)

    return resnet

def createDropoutUnit(drop_rate, layer_sizes):
    layers = []
    prev = layer_sizes[0]
    for i, features in enumerate(layer_sizes[1:-1]):
        layers.extend([
            ('fc{}'.format(i), nn.Linear(prev, features)),
            ('drop{}'.format(i), nn.Dropout(p=drop_rate)),
            ('relu{}'.format(i), nn.ReLU(inplace=True)),
        ])

        prev = features
    layers.extend([
        ('fc', nn.Linear(prev, layer_sizes[-1]))
        ])

    return nn.Sequential(OrderedDict(layers))

#########################################################################
# DYNAMIC MODEL MODIFICATION
#########################################################################

def decayLR(optim, epoch, model, history):
    decay_rate = model.decay_schedule["decay_rate"]
    decay = model.decay_schedule["decay_coeff"]
    decay_thres = model.decay_schedule["decay_thres"]
    circular_lr = model.decay_schedule["circular_lr"]

    isDecayEpoch = False
    if decay == 1:
        return optim
    elif decay_thres is not None:
        if len(history) < WINDOW_LENGTH: return optim
        if (pstdev(history[-WINDOW_LENGTH:]) < decay_thres):
            isDecayEpoch = True
            print("Decaying due to training plateau...")
    else:
        isDecayEpoch = ((epoch + 1) % decay_rate == 0)

    if isDecayEpoch:
        for params_dict in model.optim_params:
            params_dict['lr'] = decay * params_dict['lr']

    if circular_lr is not None and epoch > 0:
        print ("SHOULD NOT ENTER! CIRCULAR LR (WIP)")
        if epoch % 2 == 1:
            mult = circular_lr
        else:
            mult = 1/circular_lr
        for params_dict in model.optim_params:
            params_dict['lr'] = mult * params_dict['lr']

    return torch.optim.Adam(model.optim_params)

str_to_net = {
    'net': Net,
    'dummy': DummyNet,
    'alex': AlexFineTuned,
    'res': ResNetFineTuned,
    'res34': ResNet34FineTuned,
    'resnext': ResNextFineTuned
}

