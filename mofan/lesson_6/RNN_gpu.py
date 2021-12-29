import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data
import matplotlib.pyplot as plt

# Hyper parameters
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MINST = False

train_data = torchvision.datasets.MNIST(
    root='./minst',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINST,
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=2,
)

test_data = torchvision.datasets.MNIST(
    root='./minst',
    train=False,
    # transform=torchvision.transforms.ToTensor(),
)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.float32)[:].cuda()/255.
test_y = test_data.targets[:].cuda()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        output = self.out(r_out[:, -1, :])
        return output


rnn = RNN()
rnn.cuda()
# print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1, 28, 28).cuda()
        b_y = y.cuda()
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x.view(-1, 28, 28))
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            # accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('|Epoch:', epoch,
                  '|Train loss: %.4f' % loss.data,
                  '|Test accuracy:', accuracy)

test_output = rnn(test_x[:50].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:50], 'real number')
