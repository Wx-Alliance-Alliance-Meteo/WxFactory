"""
Code to compile the kernels
"""

from glob import glob
import importlib
import itertools
import os
import re
import shutil
from types import ModuleType

from mpi4py import MPI
import pybind11
from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension

from common import main_project_dir
from wx_mpi import SingleProcess, Conditional


proc_id_re = re.compile(r"\b\d{4,10}\b")  # At least 4 digits, surrounded by whitespace
proc_vendor_re = re.compile(r"\b(intel|amd)\b")

base_library_directory = os.path.join(main_project_dir, "lib")
base_build_directory = os.path.join(base_library_directory, "build")
base_module_dir = "wx_factory"

cpp_compile_flags = "-Wall -Wextra -shared -std=c++17 -fPIC".split(" ")
cpp_link_flags = []
cuda_compile_flags = "-arch native -O2 -shared -std=c++17 -Xcompiler -fPIC,-Wall,-Wextra".split(" ")


class wx_build_ext(build_ext):
    pass


class cuda_build_ext(wx_build_ext):
    """Define CUDA compiler and linker."""

    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        self.compiler.set_executable("compiler_so", "nvcc")
        self.compiler.set_executable("linker_so", "nvcc -shared")

        build_ext.build_extensions(self)


def _ext_name(module_name: str, kernel_type: str) -> str:
    return f"{module_name}-{kernel_type}"


def get_processor_name() -> str:
    """
    Get the name of the curent processor

    :return: The name of the processor, or empty string if the name cannot be found
    """
    cpu_info_filename = "/proc/cpuinfo"
    cpu_info_field = "model name"

    if not os.path.exists(cpu_info_filename):
        return ""

    with open(cpu_info_filename) as cpu_info:
        for line in cpu_info:
            if line.startswith(cpu_info_field):
                lc = line.split(": ")[1].lower()
                vendor = proc_vendor_re.search(lc).group()
                num = proc_id_re.search(lc).group()
                return f"{vendor}_{num}"

            if line == "":
                return ""


class WxExtension(Extension):
    """Define where to find source files, headers, and where to put the module."""

    def __init__(self, name, backend, suffix, build_ext_class, **kwargs):

        common_dir = os.path.join(base_module_dir, "definitions")
        source_dir = os.path.join(base_module_dir, name)
        source_files = glob(source_dir + f"/**/*.{suffix}", root_dir=main_project_dir, recursive=True)
        include_dirs = [pybind11.get_include(), base_module_dir]
        header_files = list(
            itertools.chain.from_iterable(
                glob(subdir + f"/**/*.{s}", root_dir=main_project_dir, recursive=True)
                for s in ["h", "hpp"]
                for subdir in [common_dir, source_dir]
            )
        )

        proc_name = get_processor_name()
        self.build_dir = os.path.join(base_build_directory, name, backend)  # Specific directory for build files
        self.build_temp = os.path.join(self.build_dir, "tmp")
        self.lib_dir = os.path.join(base_library_directory, name, proc_name, backend)  # Where the lib file will end up

        # Name of the module, relative to project root
        self.output_module = f"lib.{name}.{proc_name}.{backend}.{_ext_name(name, backend)}"

        self.build_ext_class = build_ext_class

        super().__init__(
            _ext_name(name, backend),
            source_files,
            include_dirs=include_dirs,
            depends=header_files + [__file__],
            **kwargs,
        )

    def clean(self):
        """Remove any files produced by the compilation (temporary or not)."""
        for tree in [self.build_dir, self.lib_dir]:
            for root, _, files in os.walk(tree, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))


class CppExtension(WxExtension):
    """Define compilation flags for C++."""

    def __init__(self, name, **kwargs):
        super().__init__(
            name,
            "cpp",
            "cpp",
            wx_build_ext,
            extra_compile_args=cpp_compile_flags,
            extra_link_args=cpp_link_flags,
            **kwargs,
        )


class CudaExtension(WxExtension):
    """Define compilation flags for CUDA."""

    def __init__(self, name, **kwargs):
        super().__init__(
            name,
            "cuda",
            "cu",
            cuda_build_ext,
            extra_compile_args=cuda_compile_flags,
            **kwargs,
        )


# All the extensions we will want to build for running WxFactory
_extensions: dict[str, WxExtension] = {
    _ext_name("pde", "cpp"): CppExtension("pde"),
    _ext_name("pde", "cuda"): CudaExtension("pde"),
    _ext_name("operators", "cpp"): CppExtension("operators"),
    _ext_name("operators", "cuda"): CudaExtension("operators"),
}


def compile_extension(module_name: str, kernel_type: str):
    """
    Compile the kernels

    :param module_name: Name of the module we want to compile
    :param kernel_type: Type of kernel to compile (cpp or cuda)
    """

    ext = get_extension(module_name, kernel_type)
    if not os.path.exists(ext.build_dir):
        os.makedirs(ext.build_dir, exist_ok=True)
    setup(
        ext_modules=[ext],
        script_args=["build_ext"],
        options={
            "build": {
                "build_lib": ext.lib_dir,
                "build_base": ext.build_dir,
                "build_temp": ext.build_temp,
            }
        },
        cmdclass={"build_ext": ext.build_ext_class},
    )


def compile(module_name: str, kernel_type: str, force: bool = False, comm: MPI.Comm = MPI.COMM_WORLD):
    """
    Compile the given module. This is a collective call, but only one process will actually perform the
    compilation. All processes in the given communicator must call this function.

    :param module_name: The module to compile
    :param kernel_type: Type of kernels to compile [cpp, cuda]
    :param force: Whether to force a rebuild of the module
    :param comm: Communicator used by all processes calling this function
    """
    with SingleProcess(comm) as s, Conditional(s):
        if force:
            get_extension(module_name, kernel_type).clean()

        compile_extension(module_name, kernel_type)


def load_module(module_name: str, kernel_type: str) -> ModuleType:
    """Import the given module. We don't verify here if it has been compiled.

    :param module_name: The module to compile
    :param kernel_type: Type of kernels to compile [cpp, cuda]
    :return: The imported module
    """
    module_name = _extensions[_ext_name(module_name, kernel_type)].output_module
    return importlib.import_module(module_name)


def clean_all():
    """Remove build files for every extension."""
    for _, ext in _extensions.items():
        ext.clean()


def get_extension(module_name: str, kernel_type: str) -> WxExtension:
    """Retrieve extension object from its name and kernel type."""
    return _extensions[_ext_name(module_name, kernel_type)]


__all__ = ["clean_all", "compile", "get_extension", "load_module"]
