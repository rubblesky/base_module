from model import *
import torch
from . import type
from .c_function import CFuntion
from ctypes import *

configs = [
    dict(
        test_data=[
            torch.tensor([1, 2, 3, 4., 5., 6., 7., 8., 9., 10, 11, 12]).reshape(1, 1, 3, 4),
            torch.randn(1, 1, 5, 6)
        ],
        config={
            'name': 'SliceFunction',
            'function': SliceFunction,
            'params': dict(),
            'c_forward': CFuntion(name='slice', argtypes=[c_void_p, POINTER(c_int), POINTER(c_int), c_int],
                                  restype=c_void_p),
            'forward_params': ((0, 0, 1, 2), (1, 1, 2, 3),),
            'c_forward_params': ((c_int * 4)(0, 0, 1, 2), (c_int * 4)(1, 1, 2, 3), 4)
        }
    ),
    dict(
        test_data=[
            torch.randn(1, 1, 1, 1),
            torch.randn(2, 1, 3, 4),
        ],
        config={
            'name': 'SliceFunction',
            'function': SliceFunction,
            'params': dict(),
            'c_forward': CFuntion(name='slice', argtypes=[c_void_p, POINTER(c_int), POINTER(c_int), c_int],
                                  restype=c_void_p),
            'forward_params': ((0, 0, 0, 0), (1, 1, 1, 1),),
            'c_forward_params': ((c_int * 4)(0, 0, 0, 0), (c_int * 4)(1, 1, 1, 1), 4)
        }
    ),
    dict(
        test_data=[
            torch.randn(5, 4, 9, 9),
            torch.randn(8, 8, 9, 9),
        ],
        config={
            'name': 'SliceFunction',
            'function': SliceFunction,
            'params': dict(),
            'c_forward': CFuntion(name='slice', argtypes=[c_void_p, POINTER(c_int), POINTER(c_int), c_int],
                                  restype=c_void_p),
            'forward_params': ((1, 2, 6, 1), (4, 4, 9, 6),),
            'c_forward_params': ((c_int * 4)(1, 2, 6, 1), (c_int * 4)(4, 4, 9, 6), 4)
        }
    ),
]
