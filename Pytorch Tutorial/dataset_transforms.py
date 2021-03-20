import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math

class WineDataset(Dataset):

	def __init__(self, transform = None):
		# data loading
		xy = np.loadtxt('wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
		self.x = xy[:, 1:]
		self.y = xy[:, [0]]
		self.n_samples = xy.shape[0]

		self.transform = transform

	def __getitem__(self, index):
		# data[0]
		sample = self.x[index], self.y[index]

		if self.transform:
			sample = self.transform(sample)
		
		return sample

	def __len__(self):
		# len(data)
		return self.n_samples

class ToTensor:
	def __call__(self, sample):
		inputs, targets = sample
		return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
	def __init__(self, factor):
		self.factor = factor

	def __call__(self, sample):
		inputs, targets = sample 
		inputs *= self.factor
		return inputs, targets

print('ToTensor transformation : \n\n')
dataset = WineDataset(transform = ToTensor())
X, y = dataset[0]
print(X, type(X))
print(y, type(y))
print('\n\n')

print('MulTransform : \n\n')
dataset = WineDataset(transform = MulTransform(factor = 10))
X, y = dataset[0]
print(X, type(X))
print(y, type(y))
print('\n\n')

print('Composed transformation : \n\n')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(factor = 3)])
dataset = WineDataset(transform = composed)
X, y = dataset[0]
print(X, type(X))
print(y, type(y))
print('\n\n')
