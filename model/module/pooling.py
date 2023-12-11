import torch
import torch.nn as nn


class MaxPoolingModule(nn.Module):
    def __init__(self,
                 kernel_size,
                 stride,
                 padding):
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pooling(x)
