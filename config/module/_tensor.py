from model import *
import torch
from . import type
configs = [
    dict(
        test_data = [
            torch.tensor([1,2,3,4.,5.,6.,7.,8.,9.,10,11,12]).reshape(1,3,4), 
            torch.randn(1,3,4),
             ],
        config = { 
            'name'      : 'TensorModule',
            'module'    : TensorModule,
            'params'    : dict(dims = (2,1,0)),
            'model_path': None,
            'bin_path'  : None,
            'loss'      : torch.nn.MSELoss(),
        }
    ),

]
