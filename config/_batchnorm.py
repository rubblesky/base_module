from model import *
import torch
configs = [
    dict(
        test_data = [
            torch.tensor([1,2,3,4.,5.,6.,7.,8.,9.,10,11,12]).reshape(1,1,3,4), 
            torch.tensor([1,2,3,4.,5.,6.,7.,8.,9.]).reshape(1,1,3,3),
            torch.randn(1,1,3,4),
            torch.randn(1,1,4,5),
             ],
        config = { 
            'name'      : 'BatchNormModule',
            'module'    : BatchNormModule,
            'params'    : dict(num_features = 1),
            'model_path': "pth/BatchNormModule.pth",
            'bin_path'  : "bin/BatchNormModule.bin",
            'loss'      : torch.nn.MSELoss(),
        }
    ),
]
