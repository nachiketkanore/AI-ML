# linear regression from scratch

import numpy as np
import torch

# f = w * x

# f = 2 * x

X = torch.tensor([1,2,3,4], dtype = torch.float32)
Y = torch.tensor([2,4,6,8], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# model predictions
def forward(x):
	return w * x

# loss
def loss(y, y_pred):
	return ((y_pred - y) ** 2).mean()

# gradient
# MSE = 1/N * (w * x - y) ** 2
# dJ / dw = 1/N * 2 * (w * x - y)

# def gradient(x, y, y_pred):
	# return np.dot(2 * x, y_pred - y).mean()

print(f'predictions before training, f(5) {forward(5) : .3f}')

# training
learning_rate = 0.01
n_iters = 50

for epoch in range(n_iters):
	# prediction = forward pass
	y_pred = forward(X)

	# loss
	l = loss(Y, y_pred)

	# gradients = backward pass
	# dw = gradient(X, Y, y_pred)
	l.backward() # dl / dw

	# update weights
	with torch.no_grad():
		w -= learning_rate * w.grad

	# zero gradients
	w.grad.zero_()

	if epoch % 2 == 0:
		print(f'epoch {epoch + 1}, w = {w: .3f}, loss = {l: .8f}')

print('predictions : \n\n')
for X in [100,200,300,400]:
	print(f'predictions after training, f({X}) {forward(X) : .3f}')


"""
epoch 1, w =  0.300, loss =  30.00000000
epoch 3, w =  0.772, loss =  15.66018772
epoch 5, w =  1.113, loss =  8.17471695
epoch 7, w =  1.359, loss =  4.26725292
epoch 9, w =  1.537, loss =  2.22753215
epoch 11, w =  1.665, loss =  1.16278565
epoch 13, w =  1.758, loss =  0.60698116
epoch 15, w =  1.825, loss =  0.31684780
epoch 17, w =  1.874, loss =  0.16539653
epoch 19, w =  1.909, loss =  0.08633806
epoch 21, w =  1.934, loss =  0.04506890
epoch 23, w =  1.952, loss =  0.02352631
epoch 25, w =  1.966, loss =  0.01228084
epoch 27, w =  1.975, loss =  0.00641066
epoch 29, w =  1.982, loss =  0.00334642
epoch 31, w =  1.987, loss =  0.00174685
epoch 33, w =  1.991, loss =  0.00091188
epoch 35, w =  1.993, loss =  0.00047601
epoch 37, w =  1.995, loss =  0.00024848
epoch 39, w =  1.996, loss =  0.00012971
epoch 41, w =  1.997, loss =  0.00006770
epoch 43, w =  1.998, loss =  0.00003534
epoch 45, w =  1.999, loss =  0.00001845
epoch 47, w =  1.999, loss =  0.00000963
epoch 49, w =  1.999, loss =  0.00000503
predictions : 


predictions after training, f(100)  199.941
predictions after training, f(200)  399.882
predictions after training, f(300)  599.823
predictions after training, f(400)  799.763
"""