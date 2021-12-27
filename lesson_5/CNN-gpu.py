import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch.utils.data as Data
import time

EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MINST = False

train_data = torchvision.datasets.MNIST(
    root='./minst',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINST,
)

# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,      # this code should be baned(?) in Win10
)

test_data = torchvision.datasets.MNIST(
    root='./minst',
    train=False,
    # transform=torchvision.transforms.ToTensor(),
)

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:].cuda()/255.
test_y = test_data.targets[:].cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (5, 5), (1, 1), 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)       # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)   # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

start_time = time.time()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.cuda()
        b_y = y.cuda()
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            # accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('|Epoch:', epoch,
                  '|Train loss: %.4f' % loss.data,
                  '|Test accuracy:', accuracy)

end_time = time.time()
print('Train time: ', end_time-start_time, ' s')

test_output = cnn(test_x[:50])
pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
print(pred_y, 'prediction number')
print(test_y[:50], 'real number')
