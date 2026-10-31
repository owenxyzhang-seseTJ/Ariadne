"""Optional CPU kernel; the reference NumPy/SciPy path remains available."""
import sys
import os
import numpy
from setuptools import Extension,setup

extensions=[Extension('ariadne._native',['src/ariadne/_native.cpp'],
    include_dirs=[numpy.get_include()],language='c++',optional=os.environ.get('ARIADNE_REQUIRE_NATIVE')!='1',
    extra_compile_args=['/O2','/std:c++17'] if sys.platform=='win32' else ['-O3','-std=c++17'])]
setup(ext_modules=[] if os.environ.get('ARIADNE_DISABLE_NATIVE')=='1' else extensions)
