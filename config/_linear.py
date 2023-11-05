from model import *
import torch
test_data = [torch.tensor([[1.0,2,3,100,5]]),
        torch.tensor([[1.0,2,3,863,5]]),
        torch.tensor([[1.0,2,3,4,456]]),
        torch.tensor([[235.0,2,3,4,5]]),
        torch.tensor([[1.0,1285,3,4,5]]),]
config = { 
    'name'      : 'LinearModule',
    'module'    : LinearModule,
    'params'    : dict(in_features = 5,out_features = 10),
    'model_path': "pth/LinearModule.pth",
    'bin_path'  : "bin/LinearModule.bin",
    'loss'      : torch.nn.MSELoss(),
    'data'      : test_data,
}
