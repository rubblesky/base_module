import torch
import torch.nn as nn


class ReluFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return torch.relu(x)
