import numpy as np
import torch
import math
import torch.nn as nn


def detector_region(x):
    return torch.cat((
        x[:, 46 : 66, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 78 : 98].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 109 : 129].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)


class Net(torch.nn.Module):

    def __init__(self, data_len = 65, node_size = 4096):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(data_len,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,node_size)
        self.fc3 = nn.Linear(node_size,node_size)
        self.fc4 = nn.Linear(node_size,node_size)
        self.fc5 = nn.Linear(node_size,256)
        self.fc6 = nn.Linear(256,2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        

    def forward(self, x):

        print(x.dtype)

        x = self.fc1(x)
        #x = self.relu(x)
        x = self.fc2(x)
        #x = self.relu(x)
        x = self.fc3(x)
        #x = self.relu(x)
        x = self.fc4(x)
        #x = self.relu(x)
        x = self.fc5(x)
        #x = self.relu(x)
        x = self.fc6(x)
        #output = self.sigmoid(x)

        return x
        

if __name__ == '__main__':
    print(Net())