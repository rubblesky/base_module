from ctypes import *
import torch


class CTensor(Structure):
    _fields_ = [("num_dim", c_int),
                ("shape", c_int * 10),
                ("num_features", c_int),
                ("data", POINTER(c_float))]

    _wrapper_ = [("data", POINTER(c_float))]


class TestFunction(object):
    """
    A test module to test the Ansible test runner.
    """

    def __init__(self, config):
        data = config['test_data']
        model = config['config']['function']()

        self.lib = cdll.LoadLibrary('build/linux/x86_64/release/liblibrary.so')
        self.lib.Tensor_new.argtypes = [c_int, POINTER(c_int), c_void_p]
        self.lib.Tensor_new.restype = c_void_p

        self.lib.Tensor_init.argtypes = [c_int, POINTER(c_int)]
        self.lib.Tensor_init.restype = c_void_p

        self.lib.free_tensor.argtypes = [c_void_p]

        self.lib.print_tensor.argtypes = [c_void_p]
        self.print_tensor = self.lib.print_tensor

        self.module = model
        self.data = data
        self.module.eval()
        self.c_data = [self.lib.Tensor_new(len(d.data.shape), (c_int * len(d.data.shape))(*d.data.shape), d.data_ptr())
                       for d in data]

        self.c_forward = getattr(self.lib, config['config']['c_forward'].name)
        self.c_forward.argtypes = config['config']['c_forward'].argtypes
        self.c_forward.restype = config['config']['c_forward'].restype

    # def init_func(self):
    #     # raise NotImplementedError
    #     self.c_forward = None
    #     self.c_forward.argtypes = [c_void_p, ]
    #     self.c_forward.restype = c_void_p

    def tests(self, forward_params=None):
        outputs = []
        for test in self.data:
            if forward_params is not None:
                outputs.append(self.module(test, **forward_params))
            else:
                outputs.append(self.module(test))
        return outputs

    def c_tests(self, forward_params=None):
        outputs = []
        for data in self.c_data:
            if forward_params is not None:
                self.c_forward(data, **forward_params)
            else:
                outputs.append(self.c_forward(data))
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
            c_tensor = cast(c_output[i], POINTER(CTensor)).contents
            for c, p in zip(cast(c_tensor.data, POINTER(c_float)), py_output[i].reshape(-1)):
                diff.append(abs(c - float(p)))
            diffs.append(diff)
        return diffs
