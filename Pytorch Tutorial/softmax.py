# softmax -> transforms to probability
import torch
import torch.nn as nn
import numpy as np 

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis = 0)

x = np.array([2.0, 1.0, 0.1])
y = softmax(x)
print(x)
print(y)

x = torch.tensor([2.0, 1.0, 0.1])
y = torch.softmax(x, dim = 0)
print(x)
print(y)