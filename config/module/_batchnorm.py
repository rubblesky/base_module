from model import *
import torch
from . import type
functions = {
    'build_c_module': 'build_batchnorm_module',
    'create_c_output': 'create_batchnorm_output',
    'free_c_output': 'free_batchnorm_output',
    'c_forward': 'forward_batchnorm_module',
    'free_c_module': 'free_batchnorm_module',
}
configs = [
    dict(
        test_data = [
            torch.tensor([1,2,3,4.,5.,6.,7.,8.,9.,10,11,12]).reshape(1,1,3,4),
            torch.tensor([1,2,3,4.,5.,6.,7.,8.,9.]).reshape(1,1,3,3),
            torch.randn(1,1,3,4),
            torch.randn(1,1,4,5),
             ],
        train_data = [(torch.randn(1,1,3,4),torch.randn(1,1,3,4)),],
        config = {
            'name'      : 'BatchNormModule',
            'module'    : BatchNormModule,
            'params'    : dict(num_features = 1),
            'model_path': "pth/BatchNormModule.pth",
            'bin_path'  : "bin/BatchNormModule.bin",
            'loss'      : torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
    dict(
        test_data = [

            torch.randn(1,5,3,4),
            torch.randn(1,5,4,5),
             ],
        train_data = [(torch.randn(1,5,3,4),torch.randn(1,5,3,4)),],
        config = {
            'name'      : 'BatchNormModule',
            'module'    : BatchNormModule,
            'params'    : dict(num_features = 5),
            'model_path': "pth/BatchNormModule_1.pth",
            'bin_path'  : "bin/BatchNormModule_1.bin",
            'loss'      : torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
    dict(
        test_data = [

            torch.randn(1,10,30,4),
            torch.randn(1,10,4,5),
             ],
        train_data = [(torch.randn(1,10,3,4),torch.randn(1,10,3,4)),],
        config = {
            'name'      : 'BatchNormModule',
            'module'    : BatchNormModule,
            'params'    : dict(num_features = 10),
            'model_path': "pth/BatchNormModule_2.pth",
            'bin_path'  : "bin/BatchNormModule_2.bin",
            'loss'      : torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
]
