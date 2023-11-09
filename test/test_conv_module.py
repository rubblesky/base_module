from ctypes import *
from .test_module import TestModule
import torch
class TestConvModule(TestModule):
    def __init__(self,
                 data,
                 model_path,
                 bin_path):

        c_data = [(d.data_ptr(),d.shape[1],d.shape[2]) for d in data]
        module = torch.load(model_path)
        module.eval()

        conv_module = cdll.LoadLibrary('build/linux/x86_64/release/liblibrary.so')
        conv_module.build_conv_module.argtypes = [c_char_p]
        conv_module.build_conv_module.restype = c_void_p
        c_module = conv_module.build_conv_module(c_char_p(bin_path.encode('utf-8')))
        
        
        conv_module.test_conv_module.argtypes = [c_void_p,c_void_p,c_int,c_int]
        conv_module.test_conv_module.restype = POINTER(c_float)
        c_test_func = conv_module.test_conv_module

        conv_module.free_linear_module.argtypes = [c_void_p]
        self.free_c_module = conv_module.free_linear_module
        
        conv_module.free_output.argtypes = [c_void_p]
        self.free_output = conv_module.free_output


        super(TestConvModule, self).__init__(data,c_data,module,c_module,c_test_func)
    def diff(self):
        py_output = self.tests()
        c_output = self.c_tests()
        
        diffs = []
        for i,input in enumerate(self.data):
            diff = []
            c_output[i] = cast(c_output[i],POINTER(c_float *py_output[i].shape[2] * py_output[i].shape[1] * py_output[i].shape[0] ))
            print("input: \n",input)
            print("python output:\n",py_output[i])
            print("c output:")
            for l1 in range(py_output[i].shape[0]):
                for l2 in range(py_output[i].shape[1]):
                    for l3 in range(py_output[i].shape[2]):
                        print(c_output[i].contents[l1][l2][l3],end=' ')
                        diff.append(float(abs(py_output[i][l1,l2,l3]-c_output[i].contents[l1][l2][l3])))
                    print()
                print()
            print()
            diffs.append(diff)
        return diffs
