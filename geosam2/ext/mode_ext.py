import torch
import os
import torch.utils.cpp_extension as cpp_extension

_SRC = os.path.join(os.path.dirname(__file__), "mode_ext.cpp")
mode_ext = cpp_extension.load(name="mode_ext", sources=[_SRC], extra_cflags=["-fopenmp"])
mode_except_negative_one = mode_ext.mode_except_negative_one