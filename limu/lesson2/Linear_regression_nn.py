import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# make a PyTorch data iterator
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# hyper parameters
BATCH_SIZE = 10
LR = 0.03
EPOCH = 3

data_iter = load_array((features, labels), batch_size=BATCH_SIZE)

# # ???
# print(next(iter(data_iter)))

net = nn.Sequential(
    nn.Linear(2, 1),
)

net[0].weight.data.normal_(0, 0.01)         # w = normal(0, 0.01)
net[0].bias.data.fill_(0)                   # b = 0

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

for epoch in range(EPOCH):
    for X, y in data_iter:
        loss = loss_func(net(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    l = loss_func(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
