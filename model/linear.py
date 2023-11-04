import torch
import torch.nn as nn

class LinearModule(nn.Module):
    def __init__(self, 
                 in_features:int, 
                 out_features:int
                 ):
        super(LinearModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)