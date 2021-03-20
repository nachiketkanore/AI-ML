# MNIST
# Dataloader, Transformation
# Multilayer Neural Net, activation functions
# Loss and optimizer
# Training Loop(batch training)
# Model evaluation

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size = 784 # 28 x 28
hidden_size = 100
num_classes = 10
epochs = 10
batch_size = 100
learning_rate = 2e-3

# MNIST
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)
# torch.Size([100, 1, 28, 28]) torch.Size([100])
#       100 samples each with [1 channel, 28 height, 28 width]
# 		100 labels

for i in range(6):
	plt.subplot(2, 3, i+1)
	plt.imshow(samples[i][0], cmap = 'gray')

# plt.show()

class NeuralNet(nn.Module):

	def __init__(self, input_size, hidden_size, num_classes):
		super(NeuralNet, self).__init__()

		self.l1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.l2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		# no sigmoid required since we'll use cross-entropy
		return out

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training
n_total_steps = len(train_loader)
for epoch in range(epochs):
	for i, (images, labels) in enumerate(train_loader):
		# [100, 1, 28, 28] have
		# [100, 784] want
		images = images.reshape(-1, 28 * 28).to(device)
		labels = labels.to(device)

		# forward and loss
		outputs = model(images)
		loss = criterion(outputs, labels)

		# backwards
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % 100 == 0:
			print(f'epoch = {epoch + 1} / {epochs}, step = {i + 1} / {n_total_steps}, loss = {loss.item() : .5f}')


# testing
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	for images, labels in test_loader:
		images = images.reshape(-1, 28 * 28).to(device)
		labels = labels.to(device)
		outputs = model(images)

		# value, index
		_, preds = torch.max(outputs, 1)
		n_samples += labels.shape[0]
		n_correct += (preds == labels).sum().item()

	acc = n_correct * 100.0 / n_samples
	print(f'accuracy = {acc} %')

"""
epoch = 1 / 10, step = 1 / 600, loss =  2.31881
epoch = 1 / 10, step = 101 / 600, loss =  0.29769
epoch = 1 / 10, step = 201 / 600, loss =  0.28845
epoch = 1 / 10, step = 301 / 600, loss =  0.11815
epoch = 1 / 10, step = 401 / 600, loss =  0.14551
epoch = 1 / 10, step = 501 / 600, loss =  0.23875
epoch = 2 / 10, step = 1 / 600, loss =  0.11492
epoch = 2 / 10, step = 101 / 600, loss =  0.10257
epoch = 2 / 10, step = 201 / 600, loss =  0.13676
epoch = 2 / 10, step = 301 / 600, loss =  0.06606
epoch = 2 / 10, step = 401 / 600, loss =  0.14795
epoch = 2 / 10, step = 501 / 600, loss =  0.17719
epoch = 3 / 10, step = 1 / 600, loss =  0.17103
epoch = 3 / 10, step = 101 / 600, loss =  0.05948
epoch = 3 / 10, step = 201 / 600, loss =  0.16853
epoch = 3 / 10, step = 301 / 600, loss =  0.20115
epoch = 3 / 10, step = 401 / 600, loss =  0.11546
epoch = 3 / 10, step = 501 / 600, loss =  0.08856
epoch = 4 / 10, step = 1 / 600, loss =  0.05875
epoch = 4 / 10, step = 101 / 600, loss =  0.07320
epoch = 4 / 10, step = 201 / 600, loss =  0.03057
epoch = 4 / 10, step = 301 / 600, loss =  0.10850
epoch = 4 / 10, step = 401 / 600, loss =  0.06577
epoch = 4 / 10, step = 501 / 600, loss =  0.09918
epoch = 5 / 10, step = 1 / 600, loss =  0.05484
epoch = 5 / 10, step = 101 / 600, loss =  0.06499
epoch = 5 / 10, step = 201 / 600, loss =  0.02110
epoch = 5 / 10, step = 301 / 600, loss =  0.02081
epoch = 5 / 10, step = 401 / 600, loss =  0.09354
epoch = 5 / 10, step = 501 / 600, loss =  0.04473
epoch = 6 / 10, step = 1 / 600, loss =  0.08155
epoch = 6 / 10, step = 101 / 600, loss =  0.02705
epoch = 6 / 10, step = 201 / 600, loss =  0.06050
epoch = 6 / 10, step = 301 / 600, loss =  0.02344
epoch = 6 / 10, step = 401 / 600, loss =  0.02972
epoch = 6 / 10, step = 501 / 600, loss =  0.05450
epoch = 7 / 10, step = 1 / 600, loss =  0.03962
epoch = 7 / 10, step = 101 / 600, loss =  0.15536
epoch = 7 / 10, step = 201 / 600, loss =  0.03399
epoch = 7 / 10, step = 301 / 600, loss =  0.04579
epoch = 7 / 10, step = 401 / 600, loss =  0.01458
epoch = 7 / 10, step = 501 / 600, loss =  0.03361
epoch = 8 / 10, step = 1 / 600, loss =  0.00885
epoch = 8 / 10, step = 101 / 600, loss =  0.04207
epoch = 8 / 10, step = 201 / 600, loss =  0.01575
epoch = 8 / 10, step = 301 / 600, loss =  0.07261
epoch = 8 / 10, step = 401 / 600, loss =  0.01736
epoch = 8 / 10, step = 501 / 600, loss =  0.04205
epoch = 9 / 10, step = 1 / 600, loss =  0.00297
epoch = 9 / 10, step = 101 / 600, loss =  0.03821
epoch = 9 / 10, step = 201 / 600, loss =  0.01996
epoch = 9 / 10, step = 301 / 600, loss =  0.00382
epoch = 9 / 10, step = 401 / 600, loss =  0.12274
epoch = 9 / 10, step = 501 / 600, loss =  0.01690
epoch = 10 / 10, step = 1 / 600, loss =  0.06111
epoch = 10 / 10, step = 101 / 600, loss =  0.02165
epoch = 10 / 10, step = 201 / 600, loss =  0.00463
epoch = 10 / 10, step = 301 / 600, loss =  0.02309
epoch = 10 / 10, step = 401 / 600, loss =  0.01948
epoch = 10 / 10, step = 501 / 600, loss =  0.01145
accuracy = 97.25 %
"""