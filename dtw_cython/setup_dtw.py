from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("dtw_functions_in_cython.pyx")
)