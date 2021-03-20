import torch
import torch.nn as nn
import numpy as np 
import torchvision
import torch.nn.functional as F 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
epochs = 5
batch_size = 4
learning_rate = 1e-3

# data prep
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

train_dataset = torchvision.datasets.CIFAR10(
	root = './data',
	train = True,
	download = True,
	transform = transform
	)
test_dataset = torchvision.datasets.CIFAR10(
	root = './data',
	train = False,
	download = True,
	transform = transform
	)

train_loader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size = batch_size,
	shuffle = True
	)
test_loader = torch.utils.data.DataLoader(
	test_dataset,
	batch_size = batch_size,
	shuffle = True
	)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# implement conv net
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)

for epoch in range(epochs):
	for i, (images, labels) in enumerate(train_loader):

		# original shape = [4, 3, 32, 32] => [4, 3, 1024]
		# batch_size = 4, input_channels = 3,  image_size = [32, 32], total_pixels = 32 x 32 = 1024

		images = images.to(device)
		labels = labels.to(device)

		# forward and loss
		outputs = model(images)
		loss = criterion(outputs, labels)

		# backward and updates
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 2000 == 0:
			print(f'Epoch [{epoch + 1}/ {epochs}], step [{i + 1} / {n_total_steps}], Loss {loss.item(): .6f}')

print('Training finished')

with torch.no_grad():
	n_correct = 0
	n_samples = 0
	n_class_correct = [0 for _ in range(10)]
	n_class_samples = [0 for _ in range(10)]

	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)

		_, preds = torch.max(outputs, 1)
		n_samples += labels.shape[0]
		n_correct += (preds == labels).sum().item()

		for i in range(batch_size):
			label = labels[i]
			pred = preds[i]
			if (label == pred):
				n_class_correct[label] += 1
			n_class_samples[label] += 1

	acc = 100.0 * n_correct / n_samples
	print(f'Accuracy of network = {acc} %')

	for i in range(10):
		print(f'Accuracy of class {classes[i]} = { n_class_correct[i] } / {n_class_samples[i]}')


"""
Files already downloaded and verified
Files already downloaded and verified
Epoch [1/ 5], step [2000 / 12500], Loss  2.303596
Epoch [1/ 5], step [4000 / 12500], Loss  2.273979
Epoch [1/ 5], step [6000 / 12500], Loss  2.316332
Epoch [1/ 5], step [8000 / 12500], Loss  2.255394
Epoch [1/ 5], step [10000 / 12500], Loss  2.105723
Epoch [1/ 5], step [12000 / 12500], Loss  2.161461
Epoch [2/ 5], step [2000 / 12500], Loss  2.017342
Epoch [2/ 5], step [4000 / 12500], Loss  2.178478
Epoch [2/ 5], step [6000 / 12500], Loss  1.761506
Epoch [2/ 5], step [8000 / 12500], Loss  1.967424
Epoch [2/ 5], step [10000 / 12500], Loss  1.739261
Epoch [2/ 5], step [12000 / 12500], Loss  2.110051
Epoch [3/ 5], step [2000 / 12500], Loss  1.851016
Epoch [3/ 5], step [4000 / 12500], Loss  2.128314
Epoch [3/ 5], step [6000 / 12500], Loss  2.236660
Epoch [3/ 5], step [8000 / 12500], Loss  1.383035
Epoch [3/ 5], step [10000 / 12500], Loss  1.606826
Epoch [3/ 5], step [12000 / 12500], Loss  1.803450
Epoch [4/ 5], step [2000 / 12500], Loss  1.721259
Epoch [4/ 5], step [4000 / 12500], Loss  2.006416
Epoch [4/ 5], step [6000 / 12500], Loss  2.135840
Epoch [4/ 5], step [8000 / 12500], Loss  1.236826
Epoch [4/ 5], step [10000 / 12500], Loss  1.156601
Epoch [4/ 5], step [12000 / 12500], Loss  1.031016
Epoch [5/ 5], step [2000 / 12500], Loss  2.149921
Epoch [5/ 5], step [4000 / 12500], Loss  1.551253
Epoch [5/ 5], step [6000 / 12500], Loss  1.594156
Epoch [5/ 5], step [8000 / 12500], Loss  1.228293
Epoch [5/ 5], step [10000 / 12500], Loss  2.159639
Epoch [5/ 5], step [12000 / 12500], Loss  2.415093
Training finished
Accuracy of network = 46.77 %
Accuracy of class plane = 430 / 1000
Accuracy of class car = 661 / 1000
Accuracy of class bird = 320 / 1000
Accuracy of class cat = 222 / 1000
Accuracy of class deer = 527 / 1000
Accuracy of class dog = 321 / 1000
Accuracy of class frog = 523 / 1000
Accuracy of class horse = 625 / 1000
Accuracy of class ship = 600 / 1000
Accuracy of class truck = 448 / 1000
"""