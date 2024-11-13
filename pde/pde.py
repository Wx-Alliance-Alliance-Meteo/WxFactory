import ctypes
from abc import ABC, abstractmethod
from numpy import ndarray, ascontiguousarray
from common.device import CudaDevice
from geometry import Cartesian2D, CubedSphere2D, CubedSphere3D

import pde.interface_c as interface_c
import pde.interface_cuda as interface_cuda

def get_pde(name):
    if name == "euler-cartesian":
        from pde.pde_euler_cartesian import PDEEulerCartesian

        return PDEEulerCartesian

    elif name == "euler-cubesphere":
        from pde.pde_euler_cubedsphere import PDEEulerCubedSphere

        return PDEEulerCubedSphere

class PDE(ABC):
    def __init__(self, geometry, config, device, metric):

        self.geom = geometry
        self.config = config
        self.device = device
        self.num_dim = config.num_dim
        self.nb_elements = nb_elems = config.nb_elements_total
        self.nb_var = config.nb_var

        # Predefine variables that will be set later
        self.libmodule = None
        self.pointwise_fluxes = None
        self.riemann_fluxes = None

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
        
        # Determine the cuda/cpu functions to call
        if(isinstance(device, CudaDevice)):
            self._setup_cuda()
        else:
            self._setup_cpu()

        # Determine the appropriate kernels based on the config file
        self._setup_kernels()

    def _setup_cpu(self):
        self.libmodule = interface_c
        self.pointwise_fluxes = self.pointwise_fluxes_cpu
        self.riemann_fluxes = self.riemann_fluxes_cpu

    def _setup_cuda(self):
        self.libmodule = interface_cuda
        self.pointwise_fluxes = self.pointwise_fluxes_cuda
        self.riemann_fluxes = self.riemann_fluxes_cuda

    def _setup_kernels(self):
        # Here, the retrieved functions should depend on the geometry, equations and config settings
        self.pointwise_func = self._get_pointwise_flux_function()
        self.riemann_func = self._get_riemann_flux_function()

    def _get_pointwise_flux_function(self):
        return self.libmodule.pointwise_eulercartesian_2d

    def _get_riemann_flux_function(self):
        return self.libmodule.riemann_eulercartesian_ausm_2d
    
    @abstractmethod
    def pointwise_fluxes_cuda(self):
        pass

    @abstractmethod
    def pointwise_fluxes_cpu(self):
        pass

    @abstractmethod
    def riemann_fluxes_cpu(self):
        pass

    @abstractmethod
    def riemann_fluxes_cuda(self):
        pass