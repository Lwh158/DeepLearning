import random as rd
import torch
from d2l import torch as d2l


# create a fake dataset
def sythetic_data(w, b, num_examples):
    # the data y = Xw + b with noise
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


batch_size = 10
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = sythetic_data(true_w, true_b, 1000)

# # show some examples
# print('features:', features[0], '\nlabel', labels[0])
# d2l.set_figsize()
# d2l.plt.scatter(features[:, 1].detach().numpy(),
#                 labels.detach().numpy(), 1)
# d2l.plt.show()


# divide data into batches
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    rd.shuffle(indices)     # make the data out-of-order
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        # yield: return the answer in every loop


for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# define a model
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# define loss function
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


# define the optimizer
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# hyper parameters
LR = 0.03
num_epochs = 3
# initialize the model parameters
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        # the size of l is ('batch', 1), not a constant
        l.sum().backward()
        sgd([w, b], LR, batch_size)

    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'the inaccuracy of w is {true_w - w.reshape(true_w.shape)}')
print(f'the inaccuracy of b is {true_b - b}')
