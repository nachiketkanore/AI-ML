# linear regression from scratch

import numpy as np

# f = w * x

# f = 2 * x

X = np.array([1,2,3,4], dtype = np.float32)
Y = np.array([2,4,6,8], dtype = np.float32)

w = 0.0

# model predictions
def forward(x):
	return w * x

# loss
def loss(y, y_pred):
	return ((y_pred - y) ** 2).mean()

# gradient
# MSE = 1/N * (w * x - y) ** 2
# dJ / dw = 1/N * 2 * (w * x - y)

def gradient(x, y, y_pred):
	return np.dot(2 * x, y_pred - y).mean()

print(f'predictions before training, f(5) {forward(5) : .3f}')

# training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
	# prediction = forward pass
	y_pred = forward(X)

	# loss
	l = loss(Y, y_pred)

	# gradients
	dw = gradient(X, Y, y_pred)

	# update weights
	w -= learning_rate * dw

	if epoch % 2 == 0:
		print(f'epoch {epoch + 1}, w = {w: .3f}, loss = {l: .8f}')

print(f'predictions after training, f(100) {forward(100) : .3f}')
