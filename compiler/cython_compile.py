import numpy
from setuptools import Extension
import hashlib

c_compiler_options: list[str] = [
    "-std=c++11",               # Use C++11 standard
    "-O2",                      # Enable optimization level 2
    "-march=native",            # Generate code optimized for the local machine's architecture
    "-mtune=native",            # Tune to the local machine's architecture
    "-funroll-loops",           # Unroll loops to improve vectorization opportunities
    "-ftree-vectorize",         # Enable auto-vectorization
]

include_dirs: list[str] = [
    numpy.get_include()
]

interfaces: list[str] = [
    'pde/kernels/interface.pyx'
]

sources: list[str] = interfaces + [
    "pde/kernels/euler_cartesian.cpp"
]

generated_files_to_removes: list[str] = ['./pde/kernels/interface.cpp']

def make_cython_extension() -> tuple[Extension, str, list[str], list[str]]:
    """
    Returns all the required data to compile the cython code

    :return: Return the extension to be compile, the sha1 of the options, a list of generated files to remove and a list of header to copy for intellisense
    """
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
        sha1,
        generated_files_to_removes,
        interfaces
    )
