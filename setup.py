from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name = "pde.kernels.interface",
        sources = ["pde/kernels/interface.pyx", "pde/kernels/euler_cartesian.cpp"],
        language = "c++",
        extra_compile_args = [
            "-std=c++11",               # Use C++11 standard
            "-O3",                      # Enable optimization level 3
            "-march=native",            # Generate code optimized for the local machine's architecture
            "-mtune=native",            # Tune to the local machine's architecture
            "-ffast-math",              # Allow aggressive floating-point optimizations
            "-funroll-loops",           # Unroll loops to improve vectorization opportunities
            "-ftree-vectorize",         # Enable auto-vectorization
        ],
        include_dirs=[numpy.get_include()]
    )
]

# Compiler directives for Cython
cython_compiler_directives = {
    "boundscheck": False,                 # Disable bounds checking for array access
    "wraparound": False,                  # Disable negative index checking
    "cdivision": True,                    # Enable C-style division (faster, but no division by zero check)
    "nonecheck": False,                   # Disable checks for None on function arguments (must ensure no None values)
    "initializedcheck": False,            # Avoid initializing memory to 0 for variables
    "language_level": 3,                  # Use Python 3 syntax in the Cython code
    "infer_types": True,                  # Allow Cython to infer C types (for more efficient code)
    "optimize.unpack_method_calls": True  # Optimize method calls for speed
}

setup(
    name="WxFactory",
    ext_modules=cythonize(extensions, compiler_directives=cython_compiler_directives),
    zip_safe=False,  
)