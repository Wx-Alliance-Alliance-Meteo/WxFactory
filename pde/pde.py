from abc import ABC, abstractmethod

import ctypes
from numpy import ndarray, ascontiguousarray
from mpi4py import MPI

from common.configuration import Configuration
from common.device import Device, CudaDevice
from geometry import Geometry, Cartesian2D, CubedSphere2D, CubedSphere3D, Metric2D, Metric3DTopo

try:
    import pde.interface_c as interface_c
except ModuleNotFoundError:
    if MPI.COMM_WORLD.rank == 0:
        print(f"Unable to find the interface_c module. You need to compile it")
    raise


def get_pde(name):
    if name == "euler-cartesian":
        from pde.pde_euler_cartesian import PDEEulerCartesian

        return PDEEulerCartesian

    elif name == "euler-cubesphere":
        from pde.pde_euler_cubedsphere import PDEEulerCubedSphere

        return PDEEulerCubedSphere


class PDE(ABC):
    def __init__(self, geometry: Geometry, config: Configuration, device: Device, metric: Metric2D | Metric3DTopo):

        self.geom = geometry
        self.config = config
        self.device = device

        if isinstance(geometry, Cartesian2D):
            self.num_dim = 2
            self.num_var = 4
            self.num_elements = config.num_elements_horizontal * config.num_elements_vertical
        elif isinstance(geometry, CubedSphere2D):
            self.num_dim = 2
            self.num_var = 3
            self.num_elements = config.num_elements_horizontal**2
        elif isinstance(geometry, CubedSphere3D):
            self.num_dim = 3
            self.num_var = 5
            self.num_elements = config.num_elements_horizontal**2 * config.num_elements_vertical
        else:
            raise ValueError(f"Unrecognized geometry {Geometry}")

        # Predefine variables that will be set later
        self.libmodule = None

        # Store the metric terms
        num_solpts = config.num_solpts
        self.metrics = device.xp.ones((self.num_dim**2, self.num_elements, num_solpts))
        self.sqrt_g = device.xp.ones((self.num_elements, num_solpts))

        # Store the cubed-sphere metrics if needed
        if isinstance(geometry, CubedSphere2D):
            self.metrics[0] = metric.H_contra_11
            self.metrics[1] = metric.H_contra_12
            self.metrics[2] = metric.H_contra_21
            self.metrics[3] = metric.H_contra_22

            self.sqrt_g[:] = metric.sqrtG

        elif isinstance(geometry, CubedSphere3D):
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

        # Determine the cuda/cpu functions to call
        if isinstance(device, CudaDevice):
            self._setup_cuda()
        else:
            self._setup_cpu()

        # Determine the appropriate kernels based on the config file
        self._setup_kernels()

    def _setup_cpu(self):
        self.libmodule = interface_c

    def _setup_cuda(self):
        try:
            import pde.interface_cuda as interface_cuda
        except ModuleNotFoundError:
            if MPI.COMM_WORLD.rank == 0:
                print(f"Unable to load the interface_cuda module, you need to compile it if you want to use the GPU")
            raise
        self.libmodule = interface_cuda

    def _setup_kernels(self):
        # Here, the retrieved functions should depend on the geometry, equations and config settings
        self.pointwise_func = self._get_pointwise_flux_function()
        self.riemann_func = self._get_riemann_flux_function()

    def _get_pointwise_flux_function(self):
        return self.libmodule.pointwise_eulercartesian_2d

    def _get_riemann_flux_function(self):
        return self.libmodule.riemann_eulercartesian_ausm_2d

    @abstractmethod
    def pointwise_fluxes(self):
        pass

    @abstractmethod
    def riemann_fluxes(self):
        pass
