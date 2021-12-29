import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden_1, n_hidden_2, n_out):
        super(Net, self).__init__()
        self.hidden_1 = nn.Linear(n_feature, n_hidden_1)
        self.hidden_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.predict = nn.Linear(n_hidden_2, n_out)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 10, 1)
print(net)

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_function = nn.MSELoss()

for t in range(10000):
    prediction = net(x)
    loss = loss_function(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 50 == 0:
        plt.clf()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.xlim(-1.25, 1.25)
        plt.ylim(-0.2, 1.2)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
