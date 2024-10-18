# SPDX-License-Identifier: GPL-3.0-or-later

from distutils.core import setup, Extension
from Cython.Build import cythonize
import sys



all_extensions = {
    "decoder" : Extension(
        name               = "qamreconciliation.decoder",
        sources            = ["qamreconciliation/decoder.pyx"],
    ),
    "alphabet" :  Extension(
        name = "qamreconciliation.alphabet",
        sources = ["qamreconciliation/alphabet.pyx"]
    ),
    "noisemapper" : Extension(
        name = "qamreconciliation.noisemapper",
        sources = ["qamreconciliation/noisemapper.pyx"]
    ),
    "utils" : Extension(
        name = "qamreconciliation.utils",
        sources = ["qamreconciliation/utils.pyx"]
    ),
    "matrix" : Extension(
        name = "qamreconciliation.matrix",
        sources = ["qamreconciliation/matrix.pyx"]
    ),
    "mutual_information" : Extension(
        name = "qamreconciliation.mutual_information",
        sources = ["qamreconciliation/mutual_information.pyx"]
    ),
    "bicm" : Extension(
        name = "qamreconciliation.bicm",
        sources = ["qamreconciliation/bicm.pyx"]
    ),
    "reconciliation": Extension(
        name               = "sims.reconciliation",
        sources            = ["sims/reconciliation.pyx"],
        include_dirs       = ["",
                              "qamreconciliation/"]
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
    packages=["qamreconciliation", "sims"],
    ext_modules = cythonize(ext_modules, language_level=3)
)
