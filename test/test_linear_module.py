from ctypes import *
from model import LinearModule
from .test_module import TestModule
import torch
class TestLinearModule(TestModule):
    def __init__(self,data):


        module = torch.load("./pth/LinearModule.pth")
        module.eval()

        linear_module = cdll.LoadLibrary('build/linux/x86_64/release/liblibrary.so')
        linear_module.build_linear_module.argtypes = [c_char_p,c_int,c_int]
        linear_module.build_linear_module.restype = c_void_p
        c_module = linear_module.build_linear_module(c_char_p("./bin/LinearModule.bin".encode('utf-8')),5,10)
        
        
        linear_module.test_linear_module.argtypes = [c_void_p, c_void_p]
        linear_module.test_linear_module.restype = POINTER(c_float * 10)
        c_test_func = linear_module.test_linear_module

        linear_module.free_linear_module.argtypes = [c_void_p]
        self.free_c_module = linear_module.free_linear_module
        
        linear_module.free_output.argtypes = [c_void_p]
        self.free_output = linear_module.free_output
        self.output = POINTER(c_float * 10)

        linear_module
        super(TestLinearModule, self).__init__(data,module,c_module,c_test_func)
    def diff(self):
        py_output = self.tests()
        c_output = self.c_tests()
        diffs = []
        for i,input in enumerate(self.data):
            diff = []
            print("input: ",input)
            print("python output:",py_output[i])
            print("c output:", end='')
            for f in range(10):
                print(c_output[i].contents[f],end=' ')
                diff.append(float(abs(py_output[i][0][f]-c_output[i].contents[f])))
            diffs.append(diff)
        return diffs
