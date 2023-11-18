from model import *
import torch
from . import type
functions = {
    'build_c_module': 'build_layernorm_module',
    'create_c_output': 'create_layernorm_output',
    'free_c_output': 'free_layernorm_output',
    'c_forward': 'forward_layernorm_module',
    'free_c_module': 'free_layernorm_module',
}
configs = [
    dict(
        test_data = [
            torch.tensor([1,2,3,4.,5.,6.,7.,8.,9.,10,11,12]).reshape(1,3,4), 
            torch.randn(1,1,3,4),
             ],
        train_data = [(torch.randn(1,1,3,4),torch.randn(1,1,3,4)),],
        config = { 
            'name'      : 'LayerNormModule',
            'module'    : LayerNormModule,
            'params'    : dict(normalized_shape = (3,4)),
            'model_path': "pth/LayerNormModule.pth",
            'bin_path'  : "bin/LayerNormModule.bin",
            'loss'      : torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
    dict(
        test_data = [

            torch.randn(3,5),
            torch.randn(5,4,5),
             ],
        train_data = [(torch.randn(1,1,5),torch.randn(1,1,5)),
                      (torch.randn(1,2,5),torch.randn(1,2,5)),
                      ],
        config = { 
            'name'      : 'LayerNormModule',
            'module'    : LayerNormModule,
            'params'    : dict(normalized_shape = 5),
            'model_path': "pth/LayerNormModule_1.pth",
            'bin_path'  : "bin/LayerNormModule_1.bin",
            'loss'      : torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
    dict(
        test_data = [

            torch.randn(1,1,2,3),
            torch.randn(1,1,2,3),
             ],
        train_data = [(torch.randn(1,2,3),torch.randn(1,2,3)),
                      (torch.randn(2,1,2,3),torch.randn(2,1,2,3)),
                      ],
        config = { 
            'name'      : 'LayerNormModule',
            'module'    : LayerNormModule,
            'params'    : dict(normalized_shape = (1,2,3)),
            'model_path': "pth/LayerNormModule_2.pth",
            'bin_path'  : "bin/LayerNormModule_2.bin",
            'loss'      : torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
]
