from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np



extensions = [
    Extension("qamreconciliation.decoder", ["qamreconciliation/decoder.pyx"])
]

setup(
    ext_modules=cythonize(extensions, include_path=[np.get_include()]),
    include_dirs=[np.get_include()]
)
