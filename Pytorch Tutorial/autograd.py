import torch

x = torch.randn(3, requires_grad = True)
print(x)

y = x + 2
# creates computational graph
print(y)

z = y * y * 2
print(z)

z = z.mean()
print(z)

z.backward()	# dz / dx
print(x.grad)

print('-'*50)
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():
	# work

x.requires_grad_(False)
print(x)

y = x.detach()
print(y)

with torch.no_grad():
	y = x + 2
	print(y)

print('-'*50)

weights = torch.ones(4, requires_grad = True)

for epoch in range(3):
	model_output = (weights * 3).sum()
	model_output.backward()
	print(weights.grad)
	weights.grad.zero_()