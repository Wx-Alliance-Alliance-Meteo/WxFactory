import shlex
import ctypes
import subprocess as sp
from common.device import CudaDevice

# Class to handle complex numbers 
class c_double_complex(ctypes.Structure): 
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]

class CtypeInterface:

    cudalib_file = 'pde/interface_cuda.so'
    cpplib_file = 'pde/interface_c.so'

    def __init__(self, device):
        self.device = device

        if isinstance(device, CudaDevice):
            self.lib_file = self.cudalib_file
            self.compiler = 'nvcc'
            self.compiler_args = '-shared -Xcompiler -fPIC'
            self.interface_file = 'pde/interface.cu'
        else:
            self.lib_file = self.cpplib_file
            self.compiler = 'g++'
            self.compiler_args = '-shared -fPIC'
            self.interface_file = 'pde/interface.cpp'

        # Compilation takes very little time, but ideally it's only done when needed
        self._compile()

        # Load the compiled libraries
        self._load_shared_libraries()
        
    def _compile(self):
        compile_command = shlex.split(f'{self.compiler} {self.interface_file} -o {self.lib_file} {self.compiler_args}')

        try:
            result = sp.run(compile_command, check=True, stdout=sp.PIPE, stderr=sp.PIPE)
            print(f"Compiled successfully for {self.device}")
            print(result.stdout.decode())

        except sp.CalledProcessError as e:
            print("Compilation failed:")
            print(e.stderr.decode())

    def _load_shared_libraries(self):
        self.lib = ctypes.CDLL(self.lib_file)

    def get_pointwise_flux_function(self, dtype):

        if dtype == self.device.xp.double:
            self.lib.pointwise_eulercartesian_2d_double.argtypes = [
                ctypes.POINTER(ctypes.c_double), 
                ctypes.POINTER(ctypes.c_double),  
                ctypes.POINTER(ctypes.c_double), 
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.lib.pointwise_eulercartesian_2d_double.restype = None

            return self.lib.pointwise_eulercartesian_2d_double
        
        elif dtype == self.device.xp.complex128:
            self.lib.pointwise_eulercartesian_2d_complex.argtypes = [
                ctypes.POINTER(c_double_complex), 
                ctypes.POINTER(c_double_complex),  
                ctypes.POINTER(c_double_complex), 
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.lib.pointwise_eulercartesian_2d_complex.restype = None

            return self.lib.pointwise_eulercartesian_2d_complex
        
    def get_riemann_flux_function(self, dtype):

        if dtype == self.device.xp.double:
            self.lib.riemann_eulercartesian_ausm_2d_double.argtypes = [
                ctypes.POINTER(ctypes.c_double),  
                ctypes.POINTER(ctypes.c_double),  
                ctypes.POINTER(ctypes.c_double),  
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.lib.riemann_eulercartesian_ausm_2d_double.restype = None

            return self.lib.riemann_eulercartesian_ausm_2d_double
        
        elif dtype == self.device.xp.complex128:
            self.lib.riemann_eulercartesian_ausm_2d_complex.argtypes = [
                ctypes.POINTER(c_double_complex), 
                ctypes.POINTER(c_double_complex),  
                ctypes.POINTER(c_double_complex), 
                ctypes.POINTER(c_double_complex), 
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
            ]
            self.lib.riemann_eulercartesian_ausm_2d_complex.restype = None

            return self.lib.riemann_eulercartesian_ausm_2d_complex
