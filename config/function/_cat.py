from model import *
import torch
from . import type
from .c_function import CFuntion
from ctypes import *

configs = [
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),
            torch.randn(1,2,3,4)

        ],
        config={
            'name': 'CatFunction',
            'function': CatFunction,
            'params': dict(),
            'c_forward': CFuntion(name='cat', argtypes=[c_void_p, c_void_p], restype=c_void_p),
            'forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),1),
            'c_forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),1)
        }
    ),
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),
            torch.randn(1,2,3,4)

        ],
        config={
            'name': 'CatFunction',
            'function': CatFunction,
            'params': dict(),
            'c_forward': CFuntion(name='cat', argtypes=[c_void_p, c_void_p], restype=c_void_p),
            'forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),1),
            'c_forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),1)
        }
    ),
    dict(
        test_data=[
            torch.randn(1,1,1,4),
            torch.randn(1,1,2,4)

        ],
        config={
            'name': 'CatFunction',
            'function': CatFunction,
            'params': dict(),
            'c_forward': CFuntion(name='cat', argtypes=[c_void_p, c_void_p], restype=c_void_p),
            'forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),2),
            'c_forward_params': (torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),2)
        }
    ),
]
