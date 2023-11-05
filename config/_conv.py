from model import *
import torch

test_data = [
             torch.tensor([1,2,3,4.,5.,6.,7.,8.,9.]).reshape(1,3,3),
             torch.randn(1,3,3),
             ]
config = { 
    'name'      : 'ConvModule',
    'module'    : ConvModule,
    'params'    : dict(in_channels = 1,out_channels = 1,kernel_size = (3,3),stride=1, padding=1, dilation=1, groups=1, bias=True),
    'model_path': "pth/ConvModule.pth",
    'bin_path'  : "bin/ConvModule.bin",
    'loss'      : torch.nn.MSELoss(),
    'data'      : test_data
}