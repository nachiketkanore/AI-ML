# cross - entropy -> used for multiclass classification

import torch
import torch.nn as nn
import numpy as np 

def cross_entropy(actual, pred):
	loss = -np.sum(actual * np.log(pred))
	return loss

# needs one hot encoded y
Y = np.array([1,0,0])

y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

first = cross_entropy(Y, y_pred_good)
second = cross_entropy(Y, y_pred_bad)

print(first)
print(second)

loss = nn.CrossEntropyLoss()

# doesn't need one-hot encoding
# 3 samples
Y = torch.tensor([2, 0, 1])

# n_samples x n_classes = 3 x 3
y_pred_good = torch.tensor([[0.1, 1.0, 2.1],
							[2.0, 1.0, 0.1],
							[2.0, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 2.1],
							[4.0, 1.0, 0.1],
							[5.0, 3.0, 6.1]])
first = loss(y_pred_good, Y)
second = loss(y_pred_bad, Y)

print(first)
print(second)

_, preds_one = torch.max(y_pred_good, 1)
_, preds_two = torch.max(y_pred_bad, 1)
print(preds_one)
print(preds_two)