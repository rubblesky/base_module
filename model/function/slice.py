import torch
import torch.nn as nn


class SliceFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, start, end):
        slices = tuple(slice(s, e) for s, e in zip(start, end))
        return x[slices]
