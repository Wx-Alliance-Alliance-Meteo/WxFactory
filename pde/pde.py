import ctypes
from abc import ABC, abstractmethod
from numpy import ndarray
from common.device import CudaDevice
from geometry import Cartesian2D, CubedSphere2D, CubedSphere3D


def get_pde(name):
    if name == "euler-cartesian":
        from pde.pde_euler_cartesian import PDEEulerCartesian

        return PDEEulerCartesian

    elif name == "euler-cubesphere":
        from pde.pde_euler_cubedsphere import PDEEulerCubedSphere

        return PDEEulerCubedSphere

def numpy_pointer(numpy_array, c_type):
    return numpy_array.ctypes.data_as(ctypes.POINTER(c_type))

def cupy_pointer(cupy_array, c_type):
    return ctypes.cast(cupy_array.data.ptr, ctypes.POINTER(c_type))

class PDE(ABC):
    def __init__(self, geometry, config, device, metric, dtype, c_interface):

        self.geom = geometry
        self.config = config
        self.device = device
        self.num_dim = config.num_dim
        self.nb_elements = nb_elems = config.nb_elements_total
        self.nb_var = config.nb_var
        self.c_interface = c_interface
        self.dtype = dtype

        # Store the metric terms 
        nb_solpts = config.nbsolpts
        self.metrics = device.xp.ones((config.num_dim**2, nb_elems, nb_solpts))
        self.sqrt_g = device.xp.ones((nb_elems, nb_solpts))

        # Store the cubed-sphere metrics if needed
        if isinstance(geometry, (CubedSphere2D, CubedSphere3D)):
            if config.num_dim == 2: 
                self.metrics[0] = metric.H_contra_11
                self.metrics[1] = metric.H_contra_12
                self.metrics[2] = metric.H_contra_21
                self.metrics[3] = metric.H_contra_22

            if config.num_dim == 3:
                self.metrics[0] = metric.H_contra_11
                self.metrics[1] = metric.H_contra_12
                self.metrics[2] = metric.H_contra_13
                self.metrics[3] = metric.H_contra_21
                self.metrics[4] = metric.H_contra_22
                self.metrics[5] = metric.H_contra_23
                self.metrics[6] = metric.H_contra_31
                self.metrics[7] = metric.H_contra_32
                self.metrics[8] = metric.H_contra_33

            self.sqrt_g[:] = metric.sqrtG

        if(isinstance(device, CudaDevice)):
            self._setup_cuda()
        else:
            self._setup_cpu()

        self._setup_kernels()

    def _setup_cpu(self):
        self.get_pointer = numpy_pointer

    def _setup_cuda(self):
        self.get_pointer = cupy_pointer

    def _setup_kernels(self):

        if self.dtype == self.device.xp.double:
            self.c_type = ctypes.c_double

        # Here, the retrieved functions should depend on the geometry, equations and config settings
        self.pointwise_func = self.c_interface.get_pointwise_flux_function(self.dtype)
        self.riemann_func = self.c_interface.get_riemann_flux_function(self.dtype)