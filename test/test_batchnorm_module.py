from ctypes import *
from .test_module import TestModule
import torch


class TestBatchNormModule(TestModule):
    def __init__(self,
                 data,
                 model_path,
                 bin_path):

        c_data = [(d.data_ptr(), d.shape[2], d.shape[3]) for d in data]
        module = torch.load(model_path)
        module.eval()

        c_library = cdll.LoadLibrary('build/linux/x86_64/release/liblibrary.so')
        c_library.build_batchnorm_module.argtypes = [c_char_p]
        c_library.build_batchnorm_module.restype = c_void_p
        c_module = c_library.build_batchnorm_module(c_char_p(bin_path.encode('utf-8')))

        c_library.test_batchnorm_module.argtypes = [c_void_p, c_void_p, c_int, c_int]
        c_library.test_batchnorm_module.restype = POINTER(c_float)
        c_test_func = c_library.test_batchnorm_module

        c_library.free_batchnorm_module.argtypes = [c_void_p]
        self.free_c_module = c_library.free_batchnorm_module

        c_library.free_output.argtypes = [c_void_p]
        self.free_output = c_library.free_output
        super(TestBatchNormModule, self).__init__(data, c_data, module, c_module, c_test_func)

    def diff(self):
        py_output = self.tests()
        c_output = self.c_tests()

        diffs = []
        for i, input in enumerate(self.data):
            diff = []
            c_output[i] = cast(c_output[i], POINTER(
                c_float * py_output[i].shape[3] * py_output[i].shape[2] * py_output[i].shape[1] * py_output[i].shape[
                    0]))
            print("\033[1;32;40m input: \033[0m \n", input)
            print("\033[1;32;40m python output:\033[0m \n ", py_output[i])
            print("\033[1;32;40m c output: \033[0m")
            for l0 in range(py_output[i].shape[0]):
                for l1 in range(py_output[i].shape[1]):
                    for l2 in range(py_output[i].shape[2]):
                        for l3 in range(py_output[i].shape[3]):
                            print(c_output[i].contents[l0][l1][l2][l3], end=' ')
                            diff.append(float(abs(py_output[i][l0, l1, l2, l3] - c_output[i].contents[l0][l1][l2][l3])))
                        print()
                    print()
                print()
            diffs.append(diff)
        return diffs
