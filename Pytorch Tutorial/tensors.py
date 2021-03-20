import torch

x = torch.empty(3)
# print(x) 

x = torch.rand(2,2)
print(x)

x = torch.ones(2, 2, dtype = torch.float16)
print(x)

x = torch.rand(3) * 100
y = x.pow(2)
print(x)
print(y)
print(x + y)
print(x - y)

print(x * y)

x = torch.rand(5, 3)
print(x)
print(x[1,1].item())

x = torch.rand(4,4)
print(x)
y = x.view(16)
print(y.size())
print(y)


import numpy as np 

a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b), b)

a.add_(1)
print(a)
print(b)

if torch.cuda.is_available():
	print('available')
else:
	print('not available')

x = torch.ones(5, requires_grad = True)
