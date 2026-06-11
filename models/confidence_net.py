import torch
import torch.nn as nn

class ConfidenceNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(2,32,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(32,32,3,padding=1),
            nn.ReLU(),

            nn.Conv2d(32,1,1)
        )

    def forward(self,x):
        return torch.sigmoid(self.net(x))