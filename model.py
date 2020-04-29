import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from utils import *

class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width):
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
    def __init__(self, num_classes, im_height, im_width):
        super(DummyNet, self).__init__()
        self.layer1 = nn.Linear(im_height * im_width * 3, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = self.layer1(x)
        return x

def AlexFineTuned(num_classes, im_height, im_width):
    alexnet = torchvision.models.alexnet(pretrained=True)
    for param in alexnet.parameters():
        param.requires_grad = False
    num_features = alexnet.classifier[6].in_features
    alexnet.classifier[6] = nn.Linear(num_features, num_classes)
    return alexnet

def ResNetFineTuned(num_classes, im_height, im_width):
    # finetuning - https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4
    resnet = torchvision.models.resnet18(pretrained=True)
    # for param in resnet.parameters():
    #   param.requires_grad = False
    ct = 0
    resnet.optim_params = []
    lrs = [0, 1e-3, 1e-2]
    partitions = [4, 1, 1]
    partition_assignment = partitionList(
            sum([1 for _ in resnet.named_children()]), partitions)
    for name, child in resnet.named_children():
        partition = partition_assignment[ct]
        for param_name, params in child.named_parameters():
            print(param_name)
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

def ResNextFineTuned(num_classes, im_heigh, im_width):
    resnext50_32x4d = torchvision.models.resnext50_32x4d(pretrained=True)
    ct = 0
    for name, child in resnext50_32x4d.named_children():
        ct += 1
        #if ct < 5:
        for name2, params in resnext50_32x4d.named_parameters():
            params.requires_grad = False
    print(ct)
    num_features = resnext50_32x4d.fc.in_features
    resnext50_32x4d.fc = nn.Linear(num_features, num_classes)
    return resnext50_32x4d

str_to_net = {
    'net': Net,
    'dummy': DummyNet,
    'alex': AlexFineTuned,
    'res': ResNetFineTuned,
    'resnext': ResNextFineTuned
}

