"""Build script for GeoSAM2; the package itself is configured in pyproject.toml.

This file exists to compile geosam2/ext/mode_ext.cpp, the C++ vote that
turns sampled point labels into face labels, at install time: ``pip install
-e . --no-build-isolation`` in an environment that has torch. Without the
compiled module, geosam2.ext.mode_ext falls back to compiling it on first
import, which needs a compiler and ninja at runtime.
"""

from __future__ import annotations

from setuptools import setup


def _extensions():
    try:
        from torch.utils.cpp_extension import CppExtension
    except ImportError:
        # No torch at build time: install the Python sources, keep the JIT fallback.
        return []
    return [CppExtension(
        "geosam2.ext._mode_ext",
        ["geosam2/ext/mode_ext.cpp"],
        extra_compile_args=["-fopenmp", "-O3"],
        extra_link_args=["-fopenmp"],
    )]


def _cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError:
        return {}
    return {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}


setup(ext_modules=_extensions(), cmdclass=_cmdclass())
