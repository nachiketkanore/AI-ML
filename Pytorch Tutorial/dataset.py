import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math

class WineDataset(Dataset):

	def __init__(self):
		# data loading
		xy = np.loadtxt('wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
		self.x = torch.from_numpy(xy[:, 1:])
		self.y = torch.from_numpy(xy[:, [0]])
		self.n_samples = xy.shape[0]

	def __getitem__(self, index):
		# data[0]
		return self.x[index], self.y[index]

	def __len__(self):
		# len(data)
		return self.n_samples

dataset = WineDataset()
# X, y = dataset[0]
# print(X, y)
dataloader = DataLoader(dataset = dataset, batch_size = 4, shuffle = True, num_workers = 2)

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data 
# print(features, labels)

# training loop
epochs = 10
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)

for epoch in range(epochs):
	for i, (inputs, labels) in enumerate(dataloader):
		# forward and loss

		# backward

		# updates

		if (i + 1) % 5 == 0:
			print(f'epoch = {epoch+1} / {epochs}, step = {i + 1} / {n_iterations}, inputs = {inputs.shape}, outputs = {labels.shape} ')
