import os
from typing import Optional
import hashlib
import shutil
import subprocess
from mpi4py import MPI

cpu_info_filename: str = "/proc/cpuinfo"
cpu_info_field: str = "model name"
library_path: str = "lib"
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


def get_make_file_hash() -> str:
    if not os.path.exists(make_file_path):
        raise Exception("No makefile found")

    make_file_content: str
    with open(make_file_path, "rt") as make_file:
        make_file_content = "\n".join(make_file.readlines())

    return hash(make_file_content)


def save_hash(kernel_type: str):

    lib_hash = hash(get_processor_name() + get_make_file_hash())
    kernel_lib_path = get_kernel_lib_path(kernel_type)
    os.makedirs(kernel_lib_path, exist_ok=True)

    with open(os.path.join(kernel_lib_path, "hash"), "wt") as hash_file:
        hash_file.write(lib_hash)


def load_hash(kernel_type: str) -> Optional[str]:
    kernel_hash_path = os.path.join(get_kernel_lib_path(kernel_type), "hash")
    if not os.path.exists(kernel_hash_path):
        return None

    hash_content = ""
    with open(kernel_hash_path, "rt") as hash_file:
        hash_content = hash_file.readline()

    return hash_content


def compare_hash(previous_hash: str, proc_arch: Optional[str]) -> bool:
    if proc_arch is None:
        return False

    current_hash = hash(proc_arch + get_make_file_hash())
    return previous_hash == current_hash


def clean(kernel_type: str):
    kernel_lib_path = get_kernel_lib_path(kernel_type)
    if os.path.exists(kernel_lib_path):
        shutil.rmtree(kernel_lib_path)

    subprocess.run(["make", f"clean-{kernel_type}"])


def compile(kernel_type: str, force: bool = False):
    proc_arch = get_processor_name()
    hash = load_hash(kernel_type)

    recompile: bool = hash is None or not compare_hash(hash, proc_arch)

    if recompile or force:
        clean(kernel_type)
        subprocess.run(["make", kernel_type])
        save_hash(kernel_type)

def mpi_compile(kernel_type: str, force: bool = False, comm: MPI.Comm = MPI.COMM_WORLD):
    error: None
    if comm.rank == 0:
        try:
            compile(kernel_type, force=force)
        except Exception as e:
            error = e
    
    comm.Bcast(error, root=0)

    if error is not None:
        raise error
