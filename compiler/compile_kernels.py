from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension
from glob import glob
import os
import pybind11
import shutil

__all__ = ["compile", "clean"]

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

library_directory = os.path.join(main_project_dir, "lib")
build_directory = os.path.join(library_directory, "build")
pde_directory = os.path.join(library_directory, "pde")
kernel_cpp_mirror = os.path.join(pde_directory, "interface_c.so")
kernel_cuda_mirror = os.path.join(pde_directory, "interface_cuda.so")


class default_build_ext(build_ext):
    def initialize_options(self):
        super().initialize_options()
        self.build_lib = library_directory
        self.build_base = build_directory
        self.build_temp = build_directory


class cuda_build_ext(default_build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        self.compiler.set_executable("compiler_so", "nvcc")
        self.compiler.set_executable("compiler_cxx", "nvcc")

        build_ext.build_extensions(self)

    def initialize_options(self):
        super().initialize_options()
        self.build_lib = library_directory
        self.build_base = build_directory
        self.build_temp = build_directory


extra_compiler_args_cpp = "-shared -std=c++11 -fPIC".split(" ")
extra_compiler_args_cuda = "-shared -std=c++11 -Xcompiler -fPIC".split(" ")
pybind_include = pybind11.get_include()
headers = glob("pde/**/*.h", root_dir=main_project_dir, recursive=True) + glob(
    "pde/**/*.hpp", root_dir=main_project_dir, recursive=True
)
cpp = glob("pde/**/*.cpp", root_dir=main_project_dir, recursive=True)
cuda = glob("pde/**/*.cu", root_dir=main_project_dir, recursive=True)


def get_non_mirror_lib_path(prefix: str) -> str:
    return glob(f"./lib/{prefix}*.so", root_dir=main_project_dir)[0]

def has_non_mirror_lib(prefix: str) -> bool:
    return len(glob(f"./lib/{prefix}*.so", root_dir=main_project_dir)) == 1

def clean(kernel_type: str):
    if has_non_mirror_lib(f"kernels-{kernel_type}"):
        os.remove(get_non_mirror_lib_path(f"kernels-{kernel_type}"))

    match kernel_type:
        case "cpp":
            if os.path.exists(kernel_cpp_mirror):
                os.remove(kernel_cpp_mirror)
        case "cuda":
            if os.path.exists(kernel_cuda_mirror):
                os.remove(kernel_cuda_mirror)

    if os.path.exists(build_directory):
        shutil.rmtree(build_directory)
    
    abs_mirror = os.path.join(main_project_dir, kernel_cuda_mirror)

    if os.path.exists(abs_mirror):
        os.remove(abs_mirror)



def compile(kernel_type: str):
    os.makedirs(os.path.join(build_directory, "pde"), exist_ok=True)

    prefix = f"kernels-{kernel_type}"
    match kernel_type:
        case "cpp":
            ext_cpp = Extension(prefix, cpp, include_dirs=[pybind_include], extra_compile_args=extra_compiler_args_cpp)
            setup(
                ext_modules=[ext_cpp],
                headers=headers,
                script_args=["build_ext"],
                cmdclass={"build_ext": default_build_ext},
            )

            lib = get_non_mirror_lib_path(prefix)
            os.makedirs(os.path.join(pde_directory), exist_ok=True)

            abs_mirror = os.path.join(main_project_dir, kernel_cpp_mirror)

            if os.path.exists(abs_mirror):
                os.remove(abs_mirror)

            os.symlink(os.path.join(main_project_dir, lib), abs_mirror)
        case "cuda":
            ext_cuda = Extension(
                prefix, cuda, include_dirs=[pybind_include], extra_compile_args=extra_compiler_args_cuda
            )
            setup(
                ext_modules=[ext_cuda],
                headers=headers,
                script_args=["build_ext"],
                cmdclass={"build_ext": cuda_build_ext},
            )

            lib = get_non_mirror_lib_path(prefix)
            os.makedirs(os.path.join(pde_directory), exist_ok=True)

            abs_mirror = os.path.join(main_project_dir, kernel_cuda_mirror)

            if os.path.exists(abs_mirror):
                os.remove(abs_mirror)

            os.symlink(os.path.join(main_project_dir, lib), abs_mirror)
