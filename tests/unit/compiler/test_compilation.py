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

            self.assertEqual(len(os.listdir(ext.lib_dir)), 0)
            kernels.compile(mod, "cpp")
            self.assertEqual(len(os.listdir(ext.lib_dir)), 1)

            kernels.load_module(mod, "cpp")

    def test_cpp_compilation_twice(self):
        for mod in modules:
            ext = kernels.get_extension(mod, "cpp")
            self.assertEqual(len(os.listdir(ext.lib_dir)), 0)
            kernels.compile(mod, "cpp")
            self.assertEqual(len(os.listdir(ext.lib_dir)), 1)
            kernels.compile(mod, "cpp", force=True)
            self.assertEqual(len(os.listdir(ext.lib_dir)), 1)


class CompilationGPUTestCases(cuda_test.CudaTestCases):
    def setUp(self):
        super().setUp()
        kernels.clean_all()

    def test_cuda_kernels_compilation(self):

        for mod in modules:
            ext = kernels.get_extension(mod, "cuda")
            self.assertEqual(len(os.listdir(ext.lib_dir)), 0)
            kernels.compile(mod, "cuda")
            self.assertEqual(len(os.listdir(ext.lib_dir)), 1)

            kernels.load_module(mod, "cuda")
