from ctypes import *
from model import LinearModule
from .test_module import TestModule
import torch


class TestLinearModule(TestModule):
    def __init__(self, data, model_path, bin_path):
        super(TestLinearModule, self).__init__(data, model_path, bin_path)

    def init_func(self):
        self.build_c_module = self.lib.build_linear_module
        self.create_c_output = self.lib.create_linear_output
        self.free_c_output = self.lib.free_linear_output
        self.c_forward = self.lib.forward_linear_module
        self.free_c_module = self.lib.free_linear_module




