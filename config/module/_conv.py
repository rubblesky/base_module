from model import *
import torch
from . import type

functions = {
    'build_c_module': 'build_conv_module',
    'create_c_output': 'create_conv_output',
    'free_c_output': 'free_conv_output',
    'c_forward': 'forward_conv_module',
    'free_c_module': 'free_conv_module',
}
configs = [
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 3, 4),
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9.]).reshape(1, 3, 3),
            torch.randn(1, 3, 4),
            torch.randn(1, 4, 5),
        ],
        config={
            'name': 'ConvModule',
            'module': ConvModule,
            'params': dict(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                           bias=True),
            'model_path': "pth/ConvModule.pth",
            'bin_path': "bin/ConvModule.bin",
            'loss': torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
    dict(
        test_data=[
            torch.arange(18).float().reshape(2, 3, 3),
            torch.randn(2, 3, 3),
            torch.randn(2, 3, 4),
            torch.randn(2, 4, 3),
        ],
        config={
            'name': 'ConvModule',
            'module': ConvModule,
            'params': dict(in_channels=2, out_channels=1, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                           bias=True),
            'model_path': "pth/ConvModule_2.pth",
            'bin_path': "bin/ConvModule_2.bin",
            'loss': torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9.]).reshape(1, 3, 3),
            torch.randn(1, 3, 4),
            torch.randn(1, 4, 3),
        ],
        config={
            'name': 'ConvModule',
            'module': ConvModule,
            'params': dict(in_channels=1, out_channels=3, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                           bias=True),
            'model_path': "pth/ConvModule_3.pth",
            'bin_path': "bin/ConvModule_3.bin",
            'loss': torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
    dict(
        test_data=[
            torch.randn(2, 3, 4),
            torch.randn(2, 4, 3),
        ],
        config={
            'name': 'ConvModule',
            'module': ConvModule,
            'params': dict(in_channels=2, out_channels=3, kernel_size=(3, 3), stride=2, padding=1, dilation=1, groups=1,
                           bias=True),
            'model_path': "pth/ConvModule_4.pth",
            'bin_path': "bin/ConvModule_4.bin",
            'loss': torch.nn.MSELoss(),
            'functions': functions,
            'forward_params': dict(),
        }
    ),
]
