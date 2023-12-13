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
        ],
        config={
            'name': 'AddFunction',
            'function': AddFunction,
            'params': dict(),
            'c_forward': CFuntion(name='add', argtypes=[c_void_p, c_void_p], restype=c_void_p),
            'forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(2, 2, 3, 1),),
            'c_forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(2, 2, 3, 1),)
        }
    ),
    dict(
        test_data=[
            torch.randn(1,1,1),
            torch.randn(2, 1, 3, 4),
        ],
        config={
            'name': 'AddFunction',
            'function': AddFunction,
            'params': dict(),
            'c_forward': CFuntion(name='add', argtypes=[c_void_p, c_void_p], restype=c_void_p),
            'forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(2, 2, 3, 1),),
            'c_forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(2, 2, 3, 1),)
        }
    ),
]
