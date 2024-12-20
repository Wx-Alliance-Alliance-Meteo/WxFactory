import unittest
import os

import compiler.compile_utils
import compiler.compile_kernels
import cuda_test


class CompilationTestCases(unittest.TestCase):
    def setUp(self):
        super().setUp()
        compiler.compile_utils.clean("cpp")
        compiler.compile_utils.clean("cuda")

    def test_cpp_kernels_compilation(self):
        lib_path = os.path.join("lib", "pde", "interface_c.so")
        hash_path = os.path.join("lib", "cpp", "hash")

        self.assertFalse(os.path.exists(lib_path))
        self.assertFalse(os.path.exists(hash_path))
        compiler.compile_utils.compile("cpp")
        self.assertTrue(os.path.exists(lib_path))
        self.assertTrue(os.path.exists(hash_path))

        import lib.pde.interface_c as interface_c

    def test_cpp_compilation_twice(self):
        lib_path = os.path.join("lib", "pde", "interface_c.so")
        hash_path = os.path.join("lib", "cpp", "hash")

        self.assertFalse(os.path.exists(lib_path))
        self.assertFalse(os.path.exists(hash_path))
        compiler.compile_utils.compile("cpp")
        self.assertTrue(os.path.exists(lib_path))
        self.assertTrue(os.path.exists(hash_path))
        compiler.compile_utils.compile("cpp", True)
        self.assertTrue(os.path.exists(lib_path))
        self.assertTrue(os.path.exists(hash_path))


class CompilationGPUTestCases(cuda_test.CudaTestCases):
    def setUp(self):
        super().setUp()
        compiler.compile_utils.clean("cpp")
        compiler.compile_utils.clean("cuda")

    def test_cuda_kernels_compilation(self):
        lib_path = compiler.compile_kernels.kernel_cuda_mirror
        hash_path = os.path.join(compiler.compile_utils.get_kernel_lib_path("cuda"), "hash")

        self.assertFalse(os.path.exists(lib_path))
        self.assertFalse(os.path.exists(hash_path))

        compiler.compile_utils.compile("cuda")

        self.assertTrue(os.path.exists(lib_path))
        self.assertTrue(os.path.exists(hash_path))

        import lib.pde.interface_cuda as interface_cuda
