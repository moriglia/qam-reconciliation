# SPDX-License-Identifier: GPL-3.0-or-later

from distutils.core import setup, Extension
from Cython.Build import cythonize
import sys



all_extensions = {
    "decoder" : Extension(
        name               = "decoder",
        sources            = ["qamreconciliation/decoder.pyx"],
        extra_compile_args = ["-fopenmp"],
        extra_link_args    = ["-fopenmp"]
    ),
    "alphabet" :  Extension(
        name = "alphabet",
        sources = ["qamreconciliation/alphabet.pyx"]
    ),
    "noisemapper" : Extension(
        name = "noisemapper",
        sources = ["qamreconciliation/noisemapper.pyx"]
    ),
    "utils" : Extension(
        name = "utils",
        sources = ["qamreconciliation/utils.pyx"]
    ),
    "matrix" : Extension(
        name = "matrix",
        sources = ["qamreconciliation/matrix.pyx"]
    ),
    "mutual_information" : Extension(
        name = "mutual_information",
        sources = ["qamreconciliation/mutual_information.pyx"]
    ),
    "bicm" : Extension(
        name = "bicm",
        sources = ["qamreconciliation/bicm.pyx"]
    )
}


# Parse the command-line argument
only_module = None
if '--only' in sys.argv:
    module_index = sys.argv.index('--only') + 1
    if module_index < len(sys.argv):
        only_module = sys.argv.pop(module_index)
        sys.argv.remove('--only')
        
# Select the specific module to build
if only_module:
    ext_modules = [all_extensions[only_module]]
else:
    ext_modules = list(all_extensions.values())

    
setup(
    name = "qamreconciliation",
    ext_modules = cythonize(ext_modules, language_level=3)
)
