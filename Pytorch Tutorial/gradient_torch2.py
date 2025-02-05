# design pipeline in pytorch
# 1) design model (input / output size, forward pass)
# 2) construct loss and optimizer
# 3) training loop
# - forward pass : compute predictions
# - backward pass : gradients
# - update weights

import torch
import torch.nn as nn

# f = w * x

# f = 2 * x

X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

X_test = torch.tensor([[5], [10], [20], [100]], dtype = torch.float32)
input_size = n_features
output_size = n_features
# model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):

	def __init__(self, input_dim, output_dim):
		super(LinearRegression, self).__init__()

		# define layers
		self.lin = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.lin(x)

model = LinearRegression(input_size, output_size)

# training
learning_rate = 0.01
n_iters = 10000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
	# prediction = forward pass
	y_pred = model(X)

	# loss
	l = loss(Y, y_pred)

	# gradients = backward pass
	# dw = gradient(X, Y, y_pred)
	l.backward() # dl / dw

	# update weights
	optimizer.step()

	# zero gradients
	optimizer.zero_grad()

	if epoch % 100 == 0:
		[w, b] = model.parameters()
		print(f'epoch {epoch + 1}, w = {w[0][0].item(): .3f}, loss = {l: .8f}')

print('predictions after training : \n')
for X in X_test:
	print(f'predictions after training, f({X}) {model(X).item() : .3f}')
print('\n\n')