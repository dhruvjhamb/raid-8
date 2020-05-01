import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from utils import *
from statistics import pstdev

WINDOW_LENGTH = 3

class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width, params):
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
    def __init__(self, num_classes, im_height, im_width, params):
        super(DummyNet, self).__init__()
        self.layer1 = nn.Linear(im_height * im_width * 3, num_classes)
        self.optim_params = self.parameters()
        self.decay_schedule = params['decay_schedule']

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        return x

def AlexFineTuned(num_classes, im_height, im_width, params):
    alexnet = torchvision.models.alexnet(pretrained=True)
    for param in alexnet.parameters():
        param.requires_grad = False
    num_features = alexnet.classifier[6].in_features
    alexnet.classifier[6] = nn.Linear(num_features, num_classes)
    return alexnet

def ResNetFineTuned(num_classes, im_height, im_width, params):
    # finetuning - https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4
    resnet = torchvision.models.resnet18(pretrained=True)
    return ResNetCommon(resnet, num_classes, params)

def ResNet34FineTuned(num_classes, im_height, im_width, params):
    resnet = torchvision.models.resnet34(pretrained=True)
    return ResNetCommon(resnet, num_classes, params)

def ResNextFineTuned(num_classes, im_height, im_width, params):
    resnext50_32x4d = torchvision.models.resnext50_32x4d(pretrained=True)
    return ResNetCommon(resnext50_32x4d, num_classes, params)

def ResNetCommon(resnet, num_classes, params):
    ct = 0
    resnet.optim_params = []

    if params != None:
        # Initialize learning rates
        lrs = params['lrs']
        partitions = params['partitions']
        partition_assignment = partitionList(
                sum([1 for _ in resnet.named_children()]), partitions)

        # Save other parameters
        resnet.decay_schedule = params['decay_schedule']

        for name, child in resnet.named_children():
            partition = partition_assignment[ct]
            for param_name, params in child.named_parameters():
                if lrs[partition] == 0:
                    params.requires_grad = False
                else:
                    optim_params = {'params': params}
                    optim_params['lr'] = lrs[partition]
                    resnet.optim_params.append(optim_params)
            ct += 1

    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, num_classes)
    return resnet

#########################################################################
# DYNAMIC MODEL MODIFICATION
#########################################################################

def decayLR(optim, epoch, model, history):
    decay_rate = model.decay_schedule["decay_rate"]
    decay = model.decay_schedule["decay_coeff"]
    decay_thres = model.decay_schedule["decay_thres"]

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
    return torch.optim.Adam(model.optim_params)

str_to_net = {
    'net': Net,
    'dummy': DummyNet,
    'alex': AlexFineTuned,
    'res': ResNetFineTuned,
    'res34': ResNet34FineTuned,
    'resnext': ResNextFineTuned
}

