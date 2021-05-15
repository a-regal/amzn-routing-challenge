import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(5, 128),
            nn.Sigmoid(),
            nn.Linear(128,1)
        )
    def forward(self, x):
        return self.block(x)