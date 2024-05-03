from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np



extensions = [
    Extension("qamreconciliation.decoder", ["qamreconciliation/decoder.pyx"]),
    Extension("qamreconciliation.matrix", ["qamreconciliation/matrix.pyx"])
]

setup(
    ext_modules=cythonize(extensions,
                          include_path=[np.get_include()],
                          gdb_debug=True),
    include_dirs=[np.get_include()]
)
