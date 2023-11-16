from ctypes import *
import torch


class CTensor(Structure):
    _fields_ = [("num_dim", c_int),
                ("shape", c_int * 10),
                ("num_features", c_int),
                ("data", POINTER(c_float))]

    _wrapper_ = [("data", POINTER(c_float))]


class TestModule(object):
    """
    A test module to test the Ansible test runner.
    """

    def __init__(self, data, model_path, bin_path):
        self.lib = cdll.LoadLibrary('build/linux/x86_64/release/liblibrary.so')
        self.lib.Tensor_new.argtypes = [c_int, POINTER(c_int), c_void_p]
        self.lib.Tensor_new.restype = c_void_p

        self.lib.Tensor_init.argtypes = [c_int, POINTER(c_int)]
        self.lib.Tensor_init.restype = c_void_p

        self.lib.free_tensor.argtypes = [c_void_p]

        self.lib.print_tensor.argtypes = [c_void_p]
        self.print_tensor = self.lib.print_tensor

        self.module = torch.load(model_path)
        self.data = data
        self.module.eval()
        self.c_data = [self.lib.Tensor_new(len(d.data.shape), (c_int * len(d.data.shape))(*d.data.shape), d.data_ptr())
                       for d in data]

        self.init_func()
        self.build_c_module.argtypes = [c_char_p]
        self.build_c_module.restype = c_void_p
        self.c_module = self.build_c_module(c_char_p(bin_path.encode('utf-8')))

        self.create_c_output.argtypes = [c_void_p, c_void_p]
        self.create_c_output.restype = c_void_p

        self.free_c_output.argtypes = [c_void_p]

        self.c_forward.argtypes = [c_void_p, c_void_p, c_void_p]

        self.free_c_module.argtypes = [c_void_p]

        self.output = [self.create_c_output(self.c_module, d) for d in self.c_data]

    def init_func(self):
        raise NotImplementedError
        # self.build_c_module = None
        # self.create_c_output = None
        # self.free_c_output = None
        # self.c_forward = None
        # self.free_c_module = None

    def tests(self):
        outputs = []
        for test in self.data:
            outputs.append(self.module(test))
        return outputs

    def c_tests(self):
        outputs = []
        for data, output in zip(self.c_data, self.output):
            self.c_forward(self.c_module, data, output)
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
            c_tensor = cast(c_output[i], POINTER(CTensor)).contents
            for c, p in zip(cast(c_tensor.data, POINTER(c_float)), py_output[i].reshape(-1)):
                diff.append(abs(c - float(p)))
            diffs.append(diff)
        return diffs
