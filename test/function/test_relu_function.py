from ctypes import *
from .test_fuction import TestFunction
import torch


class TestBatchNormModule(TestFunction):
    def __init__(self, config):
        super(TestBatchNormModule, self).__init__(config)

    def init_func(self):
        self.c_forward = None
        self.c_forward.argtypes = [c_void_p]
        self.c_forward.restype = c_void_p
