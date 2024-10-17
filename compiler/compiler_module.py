from compiler.cython_compile import make_cython_extension
import json
import hashlib
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
from distutils.command.build import build
import shutil
import os

compiler_directives: dict[str, bool] = {
    "boundscheck": False,                 # Disable bounds checking for array access
    "wraparound": False,                  # Disable negative index checking
    "cdivision": True,                    # Enable C-style division (faster, but no division by zero check)
    "nonecheck": False,                   # Disable checks for None on function arguments (must ensure no None values)
    "initializedcheck": False,            # Avoid initializing memory to 0 for variables
    "language_level": 3,                  # Use Python 3 syntax in the Cython code
    "infer_types": True,                  # Allow Cython to infer C types (for more efficient code)
    "optimize.unpack_method_calls": True  # Optimize method calls for speed
}

class Build(build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = 'build'
        self.build_lib = 'lib'

def get_processor_name() -> str:
    with open('/proc/cpuinfo') as cpu_info:
        while True:
            line: str = cpu_info.readline()
            if line.startswith('model name'):
                return line.split(': ')[1]

def build_librairies(force_build: bool = False):
    '''
    :param force_build : force the build even when the hash remains the same
    '''
    directives: str = json.dumps(compiler_directives, sort_keys=True)
    cython_ext, cython_sha1 = make_cython_extension()
    extensions: list[Extension] = [cython_ext]
    
    sources_sha1: str = ''
    for extension in extensions:
        for source in extension.sources:
            with open(source) as s:
                code: str = '\n'.join(s.readlines())
                sources_sha1 += hashlib.sha1(bytes(code, 'utf-8'), usedforsecurity=False).hexdigest()

    processor_name: str = get_processor_name()
    hash_content: bytes = bytes(directives + processor_name, 'utf-8')
    processor_sha1: str = hashlib.sha1(hash_content, usedforsecurity=False).hexdigest()

    final_hash: str = cython_sha1 + sources_sha1 + processor_sha1

    same: bool = not force_build
    hash_exist: bool = os.path.exists('./lib/hash')
    same = same and hash_exist

    if hash_exist:
        with open('./lib/hash') as h:
            same = same and final_hash == h.readline()

    if not same:
        if not os.path.exists('./lib'):
            os.makedirs('./lib')
        setuplog = open('./lib/log', 'w')

        sys.stdout = setuplog

        args: list[str] = sys.argv
        sys.argv = [args[0], 'build_ext']
        setup(
            name="WxFactory",
            ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
            zip_safe=False,
            cmdclass={'build': Build}
        )
        sys.argv = args
        sys.stdout = sys.__stdout__
        setuplog.close()

        shutil.rmtree('./build')
        os.remove('./pde/kernels/interface.cpp')

        with open('./lib/hash', 'w') as hash:
            hash.write(final_hash)
        shutil.copyfile('./pde/kernels/interface.pyx', './lib/pde/kernels/interface.pyx')
