from ctypes import *
from .test_module import TestModule
import torch


class TestLayerNormModule(TestModule):
    def __init__(self,
                 data,
                 model_path,
                 bin_path):
        super(TestLayerNormModule, self).__init__(data, model_path, bin_path, )

    def init_func(self):
        self.build_c_module = self.lib.build_layernorm_module
        self.create_c_output = self.lib.create_layernorm_output
        self.free_c_output = self.lib.free_layernorm_output
        self.c_forward = self.lib.forward_layernorm_module
        self.free_c_module = self.lib.free_layernorm_module
