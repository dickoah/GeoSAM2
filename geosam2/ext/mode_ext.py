"""The per-row label vote (mode_ext.cpp), compiled at install time by setup.py.

When the compiled module is missing -- the package was installed without a
compiler, or straight from the sources -- it is built on first import, as it
always was; that needs a C++ compiler and ninja on the PATH.
"""

import os

import torch  # noqa: F401  -- the compiled module links against libtorch, loaded by this import

try:
    from geosam2.ext import _mode_ext as _ext
except ImportError:
    import torch.utils.cpp_extension as cpp_extension

    _SRC = os.path.join(os.path.dirname(__file__), "mode_ext.cpp")
    _ext = cpp_extension.load(name="mode_ext", sources=[_SRC], extra_cflags=["-fopenmp"])

mode_except_negative_one = _ext.mode_except_negative_one
