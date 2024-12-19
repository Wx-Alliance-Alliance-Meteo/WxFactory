import unittest
import compiler.compile_utils
import os

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

    def test_cuda_kernels_compilation(self):
        import wx_cupy
        wx_cupy.init_wx_cupy()
        if not wx_cupy.cuda_avail:
            self.skipTest("Cuda not available")

        lib_path = os.path.join("lib", "pde", "interface_cuda.so")
        hash_path = os.path.join("lib", "cuda", "hash")
        
        self.assertFalse(os.path.exists(lib_path))
        self.assertFalse(os.path.exists(hash_path))
        compiler.compile_utils.compile("cuda")
        self.assertTrue(os.path.exists(lib_path))
        self.assertTrue(os.path.exists(hash_path))

        import lib.pde.interface_cuda as interface_cuda
    
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
