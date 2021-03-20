# design pipeline in pytorch
# 1) design model (input / output size, forward pass)
# 2) construct loss and optimizer
# 3) training loop
# - forward pass : compute predictions
# - backward pass : gradients
# - update weights

import torch
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data

X_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
print(X.shape, y.shape)
print('-' * 100)
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) Training loop
epochs = 100
for epoch in range(epochs):
	# forward and loss
	y_pred = model(X)
	loss = criterion(y_pred, y)

	# backward
	loss.backward()

	# update
	optimizer.step()
	optimizer.zero_grad()

	if epoch % 10 == 0:
		print(f'epoch : {epoch + 1}, loss = {loss.item(): .4f}')

# plot
predicted = model(X).detach().numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
