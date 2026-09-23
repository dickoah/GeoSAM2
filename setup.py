"""Build script for GeoSAM2; the package itself is configured in pyproject.toml.

The C++ extension that votes face labels (geosam2/ext/mode_ext.cpp) is still
compiled on first import for now; this file will build it at install time.
"""

from setuptools import setup

setup()
