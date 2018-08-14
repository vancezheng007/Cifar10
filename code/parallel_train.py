import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())

        return output

if __name__ == '__main__':

    # Load data
    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                             batch_size=batch_size, shuffle=True)

    model = Model(input_size, output_size)

    # GPU selection
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    for data in rand_loader:
        input = Variable(data.cuda())
        output = model(input)
        print("Outside: input size", input.size(),
              "output_size", output.size())
