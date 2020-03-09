import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
	def __init__(self, num_classes, im_height, im_width):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		self.layer1 = nn.Linear(im_height * im_width * 32, num_classes)
#		self.layer2 = nn.Linear(1000, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		#x = self.conv2(x)
		x = x.flatten(1)
		#x = F.relu(self.layer1(x))
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
