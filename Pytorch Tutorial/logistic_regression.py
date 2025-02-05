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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
 
n_samples, n_features = X.shape
print(n_samples, n_features)
print('--'*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# 1) Feature scaling - 0 mean and unit variance
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# make outputs as column vectors(tensors)
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Model
# f = wx + b, then f = sigmoid(f)
class LogisticRegression(nn.Module):

	def __init__(self, n_input_features):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(n_input_features, 1)

	def forward(self, x):
		y_pred = torch.sigmoid(self.linear(x))
		return y_pred

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training loop

epochs = 1000
for epoch in range(epochs):
	
	# forward pass and loss
	y_pred = model(X_train)
	loss = criterion(y_pred, y_train)

	# backward pass
	loss.backward()

	# updates
	optimizer.step()
	optimizer.zero_grad()

	if epoch % 10 == 0:
		print(f'epoch = {epoch + 1}, loss = {loss.item(): .4f}')

with torch.no_grad():
	y_pred = model(X_test)
	y_pred_classes = y_pred.round()
	acc = y_pred_classes.eq(y_test).sum() / y_test.shape[0]
	print(f'accuracy = {acc : .4f}')