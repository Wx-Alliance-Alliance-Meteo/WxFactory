import unittest
import os

import compiler.compile_kernels as kernels
import cuda_test

modules = ["pde"]


class CompilationTestCases(unittest.TestCase):

    def setUp(self):
        super().setUp()
        kernels.clean_all()

    def test_cpp_kernels_compilation(self):
        for mod in modules:
            ext = kernels.get_extension(mod, "cpp")

            self.assertFalse(os.path.exists(ext.lib_dir))
            kernels.compile(mod, "cpp")
            self.assertTrue(os.path.exists(ext.lib_dir))

            kernels.load_module(mod, "cpp")

    def test_cpp_compilation_twice(self):
        for mod in modules:
            ext = kernels.get_extension(mod, "cpp")
            self.assertFalse(os.path.exists(ext.lib_dir))
            kernels.compile(mod, "cpp")
            self.assertTrue(os.path.exists(ext.lib_dir))
            kernels.compile(mod, "cpp", force=True)
            self.assertTrue(os.path.exists(ext.lib_dir))


class CompilationGPUTestCases(cuda_test.CudaTestCases):
    def setUp(self):
        super().setUp()
        kernels.clean_all()

    def test_cuda_kernels_compilation(self):

        for mod in modules:
            ext = kernels.get_extension(mod, "cuda")
            self.assertFalse(os.path.exists(ext.lib_dir))
            kernels.compile(mod, "cuda")
            self.assertTrue(os.path.exists(ext.lib_dir))

            kernels.load_module(mod, "cuda")
