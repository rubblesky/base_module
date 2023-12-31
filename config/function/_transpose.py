from model import *
import torch
from . import type
from .c_function import CFuntion
from ctypes import *

configs = [
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9.]).reshape(1, 1, 3, 3),
            torch.randn(1, 1, 3, 4),
            torch.randn(1, 1, 4, 5),
        ],
        config={
            'name': 'TransposeFunction',
            'function': TransposeFunction,
            'params': dict(inplace=False),
            'c_forward': CFuntion(name='transpose',argtypes=[c_void_p,c_int,c_int],restype=c_void_p),
            'forward_params': ((1, 0),),
            'c_forward_params': (1, 0),
        }
    ),
]
