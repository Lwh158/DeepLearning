import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_function = torch.nn.MSELoss()

for t in range(10000):
    prediction = net(x)
    loss = loss_function(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# save
torch.save(net, 'net.pkl')
torch.save(net.state_dict(), 'net_params.pkl')

# restore
net2 = torch.load('net.pkl')
net3 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)
net3.load_state_dict(torch.load('net_params.pkl'))

prediction1 = net(x)
prediction2 = net2(x)
prediction3 = net3(x)
plt.subplot(131)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction1.data.numpy(), 'r-', lw=5)
plt.xlim(-1.25, 1.25)
plt.ylim(-0.2, 1.2)
plt.subplot(132)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction2.data.numpy(), 'r-', lw=5)
plt.xlim(-1.25, 1.25)
plt.ylim(-0.2, 1.2)
plt.subplot(133)
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction3.data.numpy(), 'r-', lw=5)
plt.xlim(-1.25, 1.25)
plt.ylim(-0.2, 1.2)

plt.show()

