import torch

x = torch.arange(4.0, requires_grad=True)
print('x = ', x)
print(x.grad)

y = 2*torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
print(x.grad == 4 * x)

# set the grad=0 or the grad will be added
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# not an integer
x.grad.zero_()
y = x * x
y.sum().backward()      # this code equal to the code below
# y.backward(torch.ones(len(x)))
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()          # move the value of y out the calculating chart
z = u * x               # u is viewed as a constant, not x * x
z.sum().backward()
print(x.grad == u)
