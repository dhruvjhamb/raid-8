import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn

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
	# 	param.requires_grad = False
	ct = 0
	for name, child in resnet.named_children():
	    ct += 1
	    if ct < 5:
	        for name2, params in resnet.named_parameters():
                        print(name2)
                        params.requires_grad = False
	print(ct)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)
        return resnet

str_to_net = {
	'net': Net,
	'dummy': DummyNet,
	'alex': AlexFineTuned,
	'res': ResNetFineTuned,
}

