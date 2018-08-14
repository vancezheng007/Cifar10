import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5*5 square convolution
        # kernel

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# List Parameters

# params = list(net.parameters())
# print(len(params))
# for param in params:
# print(param.size())


# # Backward

# input = Variable(torch.randn(1, 1, 32, 32))
# out = net(input)
# print(out)
# net.zero_grad()
# out.backward(torch.randn(1, 10))



# # Loss function & Backward

# out = net(input)
# print(out[0])
# criterion = nn.MSELoss()
# target = Variable(torch.arange(1, 11))  # a dummy target, for example
# target = target.view(1, -1)
# loss = criterion(out, target)
# net.zero_grad()
# loss.backward()
# print(net.conv1.bias.grad)
# print(loss)



# # Update weights

# SGD
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)

# create your optimizer
input = Variable(torch.randn(1, 1, 32, 32))
target = Variable(torch.arange(1, 11))  # a dummy target, for example
optimizer = optim.SGD(net.parameters(), lr=0.01)
#in your trainning loop:
criterion = nn.MSELoss()
optimizer.zero_grad()  # zero the gradient buffers
output = net(input)
target = target.view(1, -1)
loss = criterion(output, target)
loss.backward()
optimizer.step() # does the update
