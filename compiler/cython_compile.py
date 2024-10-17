import numpy
from setuptools import Extension
import hashlib

c_compiler_options: list[str] = [
    "-std=c++11",               # Use C++11 standard
    "-O3",                      # Enable optimization level 3
    "-march=native",            # Generate code optimized for the local machine's architecture
    "-mtune=native",            # Tune to the local machine's architecture
    "-ffast-math",              # Allow aggressive floating-point optimizations
    "-funroll-loops",           # Unroll loops to improve vectorization opportunities
    "-ftree-vectorize",         # Enable auto-vectorization
]

include_dirs: list[str] = [
    numpy.get_include()
]

sources: list[str] = [
    "pde/kernels/interface.pyx", "pde/kernels/euler_cartesian.cpp"
]

def make_cython_extension() -> tuple[Extension, str]:
    hash_content: bytes = bytes(''.join(c_compiler_options) + ''.join(include_dirs) + ''.join(sources), 'utf-8')
    sha1: str = hashlib.sha1(hash_content, usedforsecurity=False).hexdigest()
    return (
        Extension(
            name='pde.kernels.interface',
            sources=sources,
            language='c++',
            extra_compile_args=c_compiler_options,
            include_dirs=include_dirs
        ),
        sha1
    )
