
import torch
from torch import nn

# hyper parameters
in_dim=1
n_hidden_1=1
n_hidden_2=1
out_dim=1

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True)
             )
        self.layer2 = nn.Sequential(
                nn.Linear(n_hidden_1, n_hidden_2),
                nn.ReLU(True),
            )
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        # print(self.modules())
        print("children")

        for i, module in enumerate( self.children()):
            print(i, module)
        print("modules")
        for i, module in enumerate( self.modules()):
            print(i, module)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = Net(in_dim, n_hidden_1, n_hidden_2, out_dim)
