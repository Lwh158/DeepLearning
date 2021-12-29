import torch
import numpy as np

# base data in torch
x = torch.arange(12)
print(x)

Shape_x = x.shape
print(Shape_x)
print(type(x))

Numel = x.numel()
print(Numel)

X = x.reshape(3, 4)
print(X)
print(X.numel())

y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(y)

Y = torch.exp(y)
print(Y)

X = torch.arange(12, dtype=float).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
Z1 = torch.cat((X, Y), dim=0)
Z2 = torch.cat((X, Y), dim=1)
print(Z1)
print(Z2)

print(X == Y)

sum_X = X.sum()
print(sum_X)

# 注意广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
# tensor([[0],
#         [1],
#         [2]]),
#  tensor([[0, 1]])
print(a + b)

print(X[-1])
print(X[1:3])

# !!!!!!!!!!!
before = id(Y)
Y = Y + X
print(id(Y) == before)

before = id(Y)
Y[:] = X + Y
print(id(Y) == before)

before = id(X)
X += Y
print(id(X) == before)

A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))

c = torch.arange(12)
d = c.reshape(3, 4)
e = c.view(3, 4)
d[:] = 2
print(c)
e[:] = 3
print(c)
print(d, e)
