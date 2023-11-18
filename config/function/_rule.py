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
            'name': 'ReluFunction',
            'function': ReluFunction,
            'params': dict(inplace=False),
            'c_forward': CFuntion(name='relu',argtypes=[c_void_p,],restype=c_void_p),
            'forward_params': dict(),
        }
    ),
]
