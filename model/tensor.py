import torch
import torch.nn as nn
class TensorModule(nn.Module):
    def __init__(self,
                 dims) -> None:
        super().__init__()
        self.dims = dims

    def forward(self,x):
        return x.transpose(self.dims)