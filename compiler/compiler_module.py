from compiler.cython_compile import make_cython_extension
import json
import hashlib
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys
from distutils.command.build import build
import shutil
import os

cpu_info_filename: str = '/proc/cpuinfo'
cpu_info_field: str = 'model name'
build_directory: str = './build'
library_directory: str = './lib'
hash_file_location: str = os.path.join(library_directory, 'hash')
log_file_location: str = os.path.join(library_directory, 'log')

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
    """
    Class to change some build parameter such as the build directory and the final library directory
    """
    def initialize_options(self):
        super().initialize_options()
        self.build_base = build_directory
        self.build_lib = library_directory

def get_processor_name() -> str|None:
    """
    Get the name of the curent processor

    :return: The name of the processor or None if the name cannot be found
    """
    if not os.path.exists(cpu_info_filename):
        return None
    with open(cpu_info_filename) as cpu_info:
        while True:
            line: str = cpu_info.readline()
            if line.startswith(cpu_info_field):
                return line.split(': ')[1]
            if line == '':
                return None

def build_librairies(force_build: bool = False):
    '''
    Build the acceleration library that Weather Factory requires to run

    :param force_build: force the build even when the hash remains the same
    '''
    
    generated_files_to_remove: list[str] = []
    interfaces_to_copy_for_intellisense: list[str] = []
    
    directives: str = json.dumps(compiler_directives, sort_keys=True)
    cython_ext, cython_sha1, cython_files_to_remove, cython_interfces = make_cython_extension()
    extensions: list[Extension] = [cython_ext]
    
    generated_files_to_remove += cython_files_to_remove
    interfaces_to_copy_for_intellisense += cython_interfces

    sources_sha1: str = ''
    for extension in extensions:
        for source in extension.sources:
            with open(source) as s:
                code: str = '\n'.join(s.readlines())
                sources_sha1 += hashlib.sha1(bytes(code, 'utf-8'), usedforsecurity=False).hexdigest()

    processor_name: str = get_processor_name()
    hash_content: bytes = bytes(directives + (processor_name or ''), 'utf-8')
    processor_sha1: str = hashlib.sha1(hash_content, usedforsecurity=False).hexdigest()

    final_hash: str = cython_sha1 + sources_sha1 + processor_sha1

    hash_exist: bool = os.path.exists(hash_file_location)
    same_hash: bool = not force_build and processor_name is not None and hash_exist

    if hash_exist:
        with open(hash_file_location) as h:
            same_hash = same_hash and final_hash == h.readline()

    if not same_hash:
        if not os.path.exists(library_directory):
            os.makedirs(library_directory)

        try:
            setuplog = open(log_file_location, 'w')

            sys.stdout = setuplog

            args: list[str] = sys.argv
            sys.argv = [args[0], 'build_ext']
            setup(
                name="WxFactory",
                ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
                zip_safe=False,
                cmdclass={'build': Build}
            )
        finally:
            sys.argv = args
            sys.stdout = sys.__stdout__
            setuplog.close()

        shutil.rmtree(build_directory)
        for file_to_remove in generated_files_to_remove:
            os.remove(file_to_remove)

        with open(hash_file_location, 'w') as hash:
            hash.write(final_hash)

        for interface_to_copy in interfaces_to_copy_for_intellisense:
            shutil.copyfile(interface_to_copy, os.path.join(library_directory, interface_to_copy))

def build_librairies_mpi():
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    
    build_error = False
    if rank == 0:
        try:
            build_librairies()
        except Exception as e:
            build_error = e
   
    MPI.COMM_WORLD.bcast(build_error, root=0)
    if build_error:
        if rank == 0:
            raise build_error
        sys.exit(-1)
