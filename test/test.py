# Python
from ctypes import *

# # 加载.so文件
# add_so = cdll.LoadLibrary('build/linux/x86_64/release/liblibrary.so')
# add_so.add.argtypes = [c_float, c_float]
# add_so.add.restype = c_float
# # 调用函数
# result = add_so.add(c_float(1), c_float(2))
# print(result)  # 输出：3
import torch
data = torch.arange(18).reshape(2, 3, 3)
print(data)
data = data.transpose(0, 2)
print(data)