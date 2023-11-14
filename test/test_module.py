from ctypes import *


class CTensor(Structure):
    _fields_ = [("num_dim", c_char),
                ("shape", c_int * 10),
                ("num_features", c_int),
                ("data", POINTER(c_float))]


class TestModule(object):
    """
    A test module to test the Ansible test runner.
    """

    def __init__(self):
        self.lib = cdll.LoadLibrary('build/linux/x86_64/release/liblibrary.so')
        self.lib.Tensor_new.argtypes = [c_int, POINTER(c_int), c_void_p]
        self.lib.Tensor_new.restype = c_void_p

        self.lib.Tensor_init.argtypes = [c_int, POINTER(c_int)]
        self.lib.Tensor_init.restype = c_void_p

        self.lib.free_tensor.argtypes = [c_void_p]

        self.lib.print_tensor.argtypes = [c_void_p]
        self.print_tensor = self.lib.print_tensor

    def tests(self):
        outputs = []
        for test in self.data:
            outputs.append(self.module(test))
        return outputs

    def c_tests(self):
        outputs = []
        for data, output in zip(self.c_data, self.output):
            self.c_test_func(self.c_module, data, output)
            outputs.append(output)
        return outputs

    def diff(self):
        py_output = self.tests()
        c_output = self.c_tests()
        diffs = []
        for i, data in enumerate(self.data):
            diff = []
            print("input: ", data)
            print("python output:", py_output[i])
            print("c output:", end='')
            self.print_tensor(c_output[i])
            diffs.append(diff)
        return diffs
