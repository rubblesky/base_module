import torch
import torch.nn as nn


class CatFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, dim):
        return torch.cat((x, y), dim)
