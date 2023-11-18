import torch
import torch.nn as nn
class BatchNormModule(nn.Module):
    def __init__(self,
                 num_features) -> None:
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features=self.num_features)
    def forward(self,x):
        return self.bn(x)