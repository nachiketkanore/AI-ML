import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib.pyplot as plt
import pandas as pd

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

# hyper parameters
input_size = 20
hidden_size = 15
num_classes = 12
num_epochs = 80
batch_size = 100
learning_rate = 1e-3

X_train = pd.read_csv('clean/X_train.csv')
y_train = pd.read_csv('clean/y_train.csv')
X_test = pd.read_csv('clean/X_test.csv')
test_values = X_test.copy()

print('Dataframe shapes : ', X_train.shape, y_train.shape, X_test.shape)

X_train = torch.tensor(X_train.values).type(torch.float32)
y_train = torch.tensor(y_train.values).type(torch.float32)
X_test = torch.tensor(X_test.values).type(torch.float32)

print('Dataframe shapes : ', X_train.shape, y_train.shape, X_test.shape)

train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_dataset = TensorDataset(Tensor(X_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size)

examples = next(iter(train_loader))
samples, labels = examples
print(samples.shape, labels.shape)
print('==============================================\n')

examples = next(iter(test_loader))
print(examples[0].shape)
print('============================================\n\n')

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
		return out

model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
	for i, (x, y) in enumerate(train_loader):
		x = x.to(device)
		y = torch.tensor([row[0] for row in y])
		y = y.long()

		# forward
		outputs = model(x)

		# print(outputs.shape, y.shape)
		loss = criterion(outputs, y)

		# backwards
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 100 == 0:
			print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/ {n_total_steps}, loss = {loss.item():.4f}')
		
predictions = []
tot = test_values.shape[0]
got = 0
# testing
with torch.no_grad():
	# rows = 200
	rows = test_values.shape[0]
	for row in range(rows):
		x = test_values.iloc[row].values
		x = torch.tensor(x).type(torch.float32)
		outputs = model(x).numpy()

		if np.size(outputs) != 0:
			got += 1

		best = np.argmax(outputs)
		predictions.append(best)


assert(got == tot)
print(set(predictions))
out = pd.DataFrame({
	'Id' : test_values['Id'],
	'Occupation' : predictions
	})

out.to_csv('nn_out/pred', index = False)