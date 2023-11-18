from ctypes import *


class CFuntion:
    def __init__(self,
                name,
                argtypes,
                restype):
        self.name = name
        self.argtypes = argtypes
        self.restype = restype
