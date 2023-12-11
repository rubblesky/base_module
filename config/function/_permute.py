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
            torch.randn(2, 5, 3, 4),
            torch.randn(2, 9, 4, 5),
        ],
        config={
            'name': 'PermuteFunction',
            'function': PermuteFunction,
            'params': dict(inplace=False),
            'c_forward': CFuntion(name='permute',argtypes=[c_void_p,POINTER(c_int),c_int],restype=c_void_p),
            'forward_params': ((1, 2, 3, 0),),
            'c_forward_params': ((c_int * 4)(1, 2, 3, 0),4)
        }
    ),
]
