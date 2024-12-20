import os
from typing import Optional
import hashlib
import shutil
from mpi4py import MPI
import compiler.compile_kernels
import sys
from importlib.metadata import version


main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

cpu_info_filename: str = "/proc/cpuinfo"
cpu_info_field: str = "model name"
library_path: str = os.path.join(main_project_dir, "lib")
make_file_path: str = "Makefile"


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


def get_kernel_lib_path(kernel_type: str):
    return os.path.join(library_path, kernel_type)


def hash(value: str) -> str:
    return hashlib.sha1(bytes(value, "utf-8"), usedforsecurity=False).hexdigest()


def get_version_hash() -> str:
    exe_version = sys.version + version("setuptools")

    return hash(exe_version)


def generate_hash() -> str:
    return hash(get_processor_name() + get_version_hash())


def save_hash(kernel_type: str, path: str):
    lib_hash = generate_hash()
    kernel_lib_path = get_kernel_lib_path(kernel_type)
    os.makedirs(kernel_lib_path, exist_ok=True)

    with open(path, "wt") as hash_file:
        hash_file.write(lib_hash)


def load_hash(kernel_type: str) -> Optional[str]:
    kernel_hash_path = os.path.join(get_kernel_lib_path(kernel_type), "hash")
    if not os.path.exists(kernel_hash_path):
        return None

    hash_content = ""
    with open(kernel_hash_path, "rt") as hash_file:
        hash_content = hash_file.readline()

    return hash_content


def compare_hash(previous_hash: str) -> bool:
    if get_processor_name() is None:
        return False

    current_hash = generate_hash()
    return previous_hash == current_hash


def clean_kernel(kernel_type: str):
    compiler.compile_kernels.clean(kernel_type)


def clean(kernel_type: str):
    kernel_lib_path = get_kernel_lib_path(kernel_type)
    if os.path.exists(kernel_lib_path):
        shutil.rmtree(kernel_lib_path)

    clean_kernel(kernel_type)
    # TODO : Add other cleaning procedure here


def compile(kernel_type: str, force: bool = False):
    proc_arch = get_processor_name()
    hash = load_hash(kernel_type)

    recompile: bool = hash is None or not compare_hash(hash)

    if recompile or force:
        clean(kernel_type)

    compiler.compile_kernels.compile(kernel_type)
    if proc_arch is not None:
        save_hash(kernel_type, os.path.join(get_kernel_lib_path(kernel_type), "hash"))


def mpi_compile(kernel_type: str, force: bool = False, comm: MPI.Comm = MPI.COMM_WORLD):
    compilation_error = None
    if comm.rank == 0:
        try:
            compile(kernel_type, force=force)
        except Exception as e:
            compilation_error = e

    compilation_error = comm.bcast(compilation_error, root=0)

    if compilation_error is not None:
        raise compilation_error
