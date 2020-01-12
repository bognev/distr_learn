import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

Num_train = 49000
transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
cifar10_train = dset.CIFAR10('./', train=True, download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(Num_train)))
cifar10_val = dset.CIFAR10('./', train=True, download=True, transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(Num_train, 50000)))
cifar10_test = dset.CIFAR10('./', train=False, download=True, transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

USE_GPU = False

dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_every = 100

print('using device:', device)

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

def test_flatten():
    x = torch.arange(12).view(2,1,3,2)
    print('before flattening:', x)
    print('after flattening:', flatten(x))

# test_flatten()

import torch.nn.functional as F

def two_layer_fc(x, params):
    x = flatten(x)
    w1, w2 = params
    x = F.relu(x@w1)
    x = x@w2
    return x

def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros((64, 50), dtype=dtype)
    w1 = torch.zeros((50, hidden_layer_size), dtype=dtype)
    w2 = torch.zeros((hidden_layer_size, 10), dtype=dtype)
    scores = two_layer_fc(x, [w1, w2])
    print(scores.size())

# two_layer_fc_test()

def three_layer_convnet(x, params):
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    x = F.conv2d(x, conv_w1, conv_b1, padding=2)
    x = F.relu(x)
    x = F.conv2d(x, conv_w2, conv_b2, padding=1)
    x = F.relu(x)
    x = flatten(x)@fc_w + fc_b
    scores = x
    return scores


def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]
# three_layer_convnet_test()

def random_weight(shape):
    if len(shape) == 2:
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:])
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w

def zero_weight(shape):
    w = torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)
    return w

print(random_weight((3,4)))

def check_accuracy_part2(loader, model_fn, params):
    split = 'val' if loader.dataset.train else  'test'
    print('Checking accuracy on the %s split'%split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct)/num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

def train_part2(model_fn, params, learning_rate):
    for t, (x,y) in enumerate(loader_train):
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)
        loss.backward()
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()

hidden_layer_size = 4000
learning_rate = 1e-2

w1 = random_weight((3 * 32 * 32, hidden_layer_size))
w2 = random_weight((hidden_layer_size, 10))

# train_part2(two_layer_fc, [w1, w2], learning_rate)

learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

# H' = 1 + (H + 2 * pad - HH) / stride

conv_w1 = random_weight((channel_1, 3, 5, 5))
conv_b1 = random_weight((channel_1, ))
conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = random_weight((channel_2, ))
fc_w = random_weight((channel_2*32*32, 10))
fc_b = random_weight((10,))

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
# train_part2(three_layer_convnet, params, learning_rate)

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TwoLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores

def test_TwoLayerFC():
    input_size = 50
    x = torch.zeros((64, input_size), dtype=dtype)
    model = TwoLayerFC(input_size, 42, 10)
    scores = model(x)
    print(scores.size())
# test_TwoLayerFC()

class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super(ThreeLayerConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        self.fc = nn.Linear(channel_2 * 32 *32, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        scores = None
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = flatten(x)
        scores = self.fc(x)
        return scores

def test_ThreeLayerConvNet():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
    scores = model(x)
    print("test_ThreeLayerConvNet", scores.size())  # you should see [64, 10]
# test_ThreeLayerConvNet()

def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (y == preds).sum()
            num_samples += preds.size(0)
        acc = float(num_correct)/num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_part34(model, optimizer, epochs=1):
    model = model.to(device=device)
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t%print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()

hidden_layer_size = 4000
learning_rate = 1e-2
model = TwoLayerFC(3*32*32, hidden_layer_size, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# train_part34(model, optimizer)

learning_rate = 3e-3
channel_1 = 32
channel_2 = 16
model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# train_part34(model, optimizer)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

hidden_layer_size = 4000
learning_rate = 1e-2
model = nn.Sequential(
    Flatten(),
    nn.Linear(3*32*32, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, 10)
)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
# train_part34(model, optimizer)

learning_rate = 3e-3
in_channel = 3
channel_1 = 32
channel_2 = 16
num_classes= 10
model = nn.Sequential(
    nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(channel_2 * 32 * 32, num_classes)
)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
train_part34(model, optimizer)