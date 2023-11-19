import torch
import torch.nn as nn


class PermuteFunction(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x:torch.Tensor,dims):
        return x.permute(dims)
