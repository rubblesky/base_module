import torch
import torch.nn as nn
class LayerNormModule(nn.Module):
    def __init__(self,
                 normalized_shape) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.ln = nn.LayerNorm(normalized_shape=self.normalized_shape)
    def forward(self,x):
        return self.ln(x)