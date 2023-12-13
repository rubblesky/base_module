import torch
import torch.nn as nn


class AddFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y
