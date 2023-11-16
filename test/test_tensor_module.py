from ctypes import *
from .test_module import TestModule
import torch
class TestTensorModule(TestModule):
    def __init__(self,
                 data,
                 model_path = None,
                 bin_path = None):
        c_data = [(d.data.ptr,) for d in data]