import torch
import torch.nn as nn


class TransposeFunction(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x:torch.Tensor,dim0,dim1):
        return x.transpose(dim0,dim1)
