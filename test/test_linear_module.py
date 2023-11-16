from ctypes import *
from model import LinearModule
from .test_module import TestModule
import torch


class TestLinearModule(TestModule):
    def __init__(self, data, model_path, bin_path):
        super(TestLinearModule, self).__init__()

        self.module = torch.load(model_path)
        self.data = data
        self.module.eval()
        self.c_data = [self.lib.Tensor_new(len(d.data.shape), (c_int * len(d.data.shape))(*d.data.shape), d.data_ptr()) for d in data]

        self.lib.build_linear_module.argtypes = [c_char_p]
        self.lib.build_linear_module.restype = c_void_p
        self.c_module = self.lib.build_linear_module(c_char_p(bin_path.encode('utf-8')))

        self.lib.create_linear_output.argtypes = [c_void_p, c_void_p]
        self.lib.create_linear_output.restype = c_void_p
        self.lib.free_linear_output.argtypes = [c_void_p]
        self.free_output = self.lib.free_linear_output

        self.lib.forward_linear_module.argtypes = [c_void_p, c_void_p, c_void_p]
        self.c_test_func = self.lib.forward_linear_module

        self.lib.free_linear_module.argtypes = [c_void_p]
        self.free_c_module = self.lib.free_linear_module

        self.lib.free_linear_output.argtypes = [c_void_p]

        self.output = [self.lib.create_linear_output(self.c_module, d) for d in self.c_data]




    # def diff(self):
    #     py_output = self.tests()
    #     c_output = self.c_tests()
    #     diffs = []
    #     for i, input in enumerate(self.data):
    #         diff = []
    #         print("input: ", input)
    #         print("python output:", py_output[i])
    #         print("c output:", end='')
    #         for f in range(10):
    #             print(c_output[i].contents[f], end=' ')
    #             diff.append(float(abs(py_output[i][0][f] - c_output[i].contents[f])))
    #         diffs.append(diff)
    #     return diffs
