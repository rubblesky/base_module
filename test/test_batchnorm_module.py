from ctypes import *
from .test_module import TestModule
import torch


class TestBatchNormModule(TestModule):
    def __init__(self,
                 data,
                 model_path,
                 bin_path):
        super(TestBatchNormModule, self).__init__(data, model_path, bin_path,)

    def init_func(self):
        self.build_c_module = self.lib.build_batchnorm_module
        self.create_c_output = self.lib.create_batchnorm_output
        self.free_c_output = self.lib.free_batchnorm_output
        self.c_forward = self.lib.forward_batchnorm_module
        self.free_c_module = self.lib.free_batchnorm_module

    # def diff(self):
    #     py_output = self.tests()
    #     c_output = self.c_tests()
    #
    #     diffs = []
    #     for i, input in enumerate(self.data):
    #         diff = []
    #         c_output[i] = cast(c_output[i], POINTER(
    #             c_float * py_output[i].shape[3] * py_output[i].shape[2] * py_output[i].shape[1] * py_output[i].shape[
    #                 0]))
    #         print("\033[1;32;40m input: \033[0m \n", input)
    #         print("\033[1;32;40m python output:\033[0m \n ", py_output[i])
    #         print("\033[1;32;40m c output: \033[0m")
    #         for l0 in range(py_output[i].shape[0]):
    #             for l1 in range(py_output[i].shape[1]):
    #                 for l2 in range(py_output[i].shape[2]):
    #                     for l3 in range(py_output[i].shape[3]):
    #                         print(c_output[i].contents[l0][l1][l2][l3], end=' ')
    #                         diff.append(float(abs(py_output[i][l0, l1, l2, l3] - c_output[i].contents[l0][l1][l2][l3])))
    #                     print()
    #                 print()
    #             print()
    #         diffs.append(diff)
    #     return diffs
