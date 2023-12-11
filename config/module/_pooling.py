from model import *
import torch
from . import type

configs = [
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 3, 4),
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9.]).reshape(1, 3, 3),
            torch.randn(1, 3, 4),
            torch.randn(1, 4, 5),],
        config={
            'name': 'PoolingModule',
            'module': MaxPoolingModule,
            'params': dict(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            'model_path': "pth/MaxPoolingModule.pth",
            'bin_path': "bin/MaxPoolingModule.bin",
            'loss': torch.nn.MSELoss(),
            'functions': {
                'build_c_module': 'build_pooling_module',
                'create_c_output': 'create_pooling_output',
                'free_c_output': 'free_pooling_output',
                'c_forward': 'forward_maxpooling_module',
                'free_c_module': 'free_pooling_module',
            },
            'forward_params': dict(),
        }
    ),
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 3, 4),
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9.]).reshape(1, 3, 3),
            torch.randn(1, 3, 4),
            torch.randn(1, 4, 5),],
        config={
            'name': 'PoolingModule',
            'module': MaxPoolingModule,
            'params': dict(kernel_size=(2, 5), stride=(1, 1), padding=(1, 2)),
            'model_path': "pth/MaxPoolingModule.pth",
            'bin_path': "bin/MaxPoolingModule.bin",
            'loss': torch.nn.MSELoss(),
            'functions': {
                'build_c_module': 'build_pooling_module',
                'create_c_output': 'create_pooling_output',
                'free_c_output': 'free_pooling_output',
                'c_forward': 'forward_maxpooling_module',
                'free_c_module': 'free_pooling_module',
            },
            'forward_params': dict(),
        }
    ),
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 3, 4),
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9.]).reshape(1, 3, 3),
            torch.randn(1, 3, 4),
            torch.randn(1, 4, 5),],
        config={
            'name': 'PoolingModule',
            'module': MaxPoolingModule,
            'params': dict(kernel_size=(3, 1), stride=(1, 2), padding=(1, 0)),
            'model_path': "pth/MaxPoolingModule.pth",
            'bin_path': "bin/MaxPoolingModule.bin",
            'loss': torch.nn.MSELoss(),
            'functions': {
                'build_c_module': 'build_pooling_module',
                'create_c_output': 'create_pooling_output',
                'free_c_output': 'free_pooling_output',
                'c_forward': 'forward_maxpooling_module',
                'free_c_module': 'free_pooling_module',
            },
            'forward_params': dict(),
        }
    ),
]
