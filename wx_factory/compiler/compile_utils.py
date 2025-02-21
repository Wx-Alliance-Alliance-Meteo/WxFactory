"""
Utilities to use to compile code
"""

import hashlib
from importlib.metadata import version
import os
import shutil
import sys
from typing import Optional

from mpi4py import MPI

from common import main_project_dir
from . import compile_kernels
from wx_mpi import SingleProcess, Conditional


cpu_info_filename: str = "/proc/cpuinfo"
cpu_info_field: str = "model name"
library_path: str = os.path.join(main_project_dir, "lib")


def get_processor_name() -> Optional[str]:
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
                return line.split(": ")[1]
            if line == "":
                return None


def get_kernel_lib_path(kernel_type: str) -> str:
    """
    Get the kernel library path

    :param kernel_type: Type of kernel
    :return: Kernel library path
    """
    return os.path.join(library_path, kernel_type)


def hash(value: str) -> str:
    """
    Hash a string

    :return: Hashed string
    """
    return hashlib.sha1(bytes(value, "utf-8"), usedforsecurity=False).hexdigest()


def get_version_hash() -> str:
    """
    Get the current version hash

    :return: Version hash
    """
    exe_version = sys.version + version("setuptools")

    return hash(exe_version)


def generate_hash() -> str:
    """
    Generate the current system hash

    :return: System hash
    """
    return hash(get_processor_name() + get_version_hash())


def save_hash(kernel_type: str):
    """
    Save a hash

    :param kernel_type: Type of kernels to use for saving
    """
    lib_hash = generate_hash()
    kernel_lib_path = get_kernel_lib_path(kernel_type)
    os.makedirs(kernel_lib_path, exist_ok=True)

    with open(os.path.join(kernel_lib_path, "hash"), "wt") as hash_file:
        hash_file.write(lib_hash)


def load_hash(kernel_type: str) -> Optional[str]:
    """
    Load a hash

    :param kernel_type: Type of kernels to use for loading
    :return: Existing hash if any
    """
    kernel_hash_path = os.path.join(get_kernel_lib_path(kernel_type), "hash")
    if not os.path.exists(kernel_hash_path):
        return None

    hash_content = ""
    with open(kernel_hash_path, "rt") as hash_file:
        hash_content = hash_file.readline()

    return hash_content


def compare_hash(previous_hash: str) -> bool:
    """
    Compare a hash with the actual hash

    :param previous_hash: Other hash to compare
    :return: True if both hashes match
    """
    if get_processor_name() is None:
        return False

    current_hash = generate_hash()
    return previous_hash == current_hash


def clean(kernel_type: str):
    """
    Clean the compilation path

    :param kernel_type: Type of kernels to clean
    """
    kernel_lib_path = get_kernel_lib_path(kernel_type)
    if os.path.exists(kernel_lib_path):
        shutil.rmtree(kernel_lib_path)

    compile_kernels.clean(kernel_type)
    # TODO : Add other cleaning procedure here


def compile(kernel_type: str, force: bool = False):
    """
    Compile the compilable module

    :param kernel_type: Type of kernels to compile
    :param force: Force a rebuild of the module
    """
    proc_arch = get_processor_name()
    hash = load_hash(kernel_type)

    recompile: bool = hash is None or not compare_hash(hash)

    if recompile or force:
        clean(kernel_type)

    compile_kernels.compile(kernel_type)
    if proc_arch is not None:
        save_hash(kernel_type)


def mpi_compile(kernel_type: str, force: bool = False, comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Compile the compilable module with MPI (not all PEs should compile)

    :param kernel_type: Type of kernels to compile
    :param force: Force a rebuild of the module
    :param comm: Communicator to use to perform the build
    """
    with SingleProcess(comm) as s, Conditional(s):
        compile(kernel_type, force=force)
