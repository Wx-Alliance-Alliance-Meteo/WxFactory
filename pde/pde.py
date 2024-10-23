from abc import ABC, abstractmethod
from numpy import ndarray
from common.device import CudaDevice
from geometry import Cartesian2D

def get_pde(name):
    if name == "euler-cartesian":
        from pde.pde_euler_cartesian import PDEEulerCartesian
        return PDEEulerCartesian

    elif name == "euler-cubesphere":
        from pde.pde_euler_cubedsphere import PDEEulerCubedSphere
        return PDEEulerCubedSphere


def load_cuda_module(filename, device):
    import os

    kernel_file = os.path.join(
        os.path.dirname(__file__), "kernels", filename)

    with open(kernel_file, 'r') as f:
        kernel_code = f.read()

    return device.xp.RawModule(code=kernel_code)


class PDE(ABC):
    def __init__(self, geometry, config, device):
        self.geom = geometry
        self.config = config
        self.device = device
        self.num_dim = config.num_dim
        self.nb_elements = config.nb_elements_total
        self.nb_var = config.nb_var

        # Set the CUDA indices to None
        if isinstance(self.device, CudaDevice):
            self._setup_cuda()
        else:
            self._setup_cpu()

    def _setup_cpu(self):
        self.pointwise_fluxes = self.pointwise_fluxes_cpu
        self.riemann_fluxes = self.riemann_fluxes_cpu

        self._setup_cpu_kernels()

    def _setup_cuda(self):
        self.pointwise_fluxes = self.pointwise_fluxes_cuda
        self.riemann_fluxes = self.riemann_fluxes_cuda
        
        # Load the CUDA raw modules where kernels are located
        self.boundary_module = load_cuda_module("boundary_flux.cu", self.device)
        self.pointwise_module = load_cuda_module("pointwise_flux.cu", self.device)
        self.riemann_module = load_cuda_module("riemann_flux.cu", self.device)
        
        self._setup_cuda_indices()
        self._setup_cuda_sizes()
        self._setup_cuda_kernels()

    def _setup_cuda_sizes(self):
        self._setup_cuda_num_threads()
        
        self.nthread_per_block = (256, )

        def compute_blocks(num_threads):
            tpb = self.nthread_per_block[0]
            return ((num_threads + tpb - 1) // tpb, )

        self.blocks_pointwise = compute_blocks(self.nthread_pointwise)

        self.blocks_riem_x1 = compute_blocks(self.nthread_riem_x1)
        self.blocks_riem_x2 = compute_blocks(self.nthread_riem_x2)
        self.blocks_riem_x3 = compute_blocks(self.nthread_riem_x3)

        self.blocks_bound_x1 = compute_blocks(self.nthread_bound_x1)
        self.blocks_bound_x2 = compute_blocks(self.nthread_bound_x2)
        self.blocks_bound_x3 = compute_blocks(self.nthread_bound_x3)

        self.riem_stride = self.nb_elements * self.config.nbsolpts * 2
    

    def _setup_cuda_num_threads(self):
        """Specify the number of threads needed for each kernel call"""

        # Pointwise fluxes are a single call
        self.nthread_pointwise = self.nb_elements * self.config.nbsolpts_elem

        # Riemann calls and boundary calls are done per dimension
        self.nthread_riem_x1 = len(self.indxl)
        self.nthread_riem_x2 = len(self.indyl)
        self.nthread_riem_x3 = len(self.indzl)

        self.nthread_bound_x1 = len(self.indbx)
        self.nthread_bound_x2 = len(self.indby)
        self.nthread_bound_x3 = len(self.indbz)


    @abstractmethod
    def _setup_cuda_kernels(self):
        """Determine the spefic kernels to run on the GPU"""
        pass

    @abstractmethod
    def _setup_cpu_kernels(self):
        """Determine the specific kernels to run on the CPU"""
        pass

    def _kernel_call(self, kernel, num_blocks, *args):  
        return kernel(num_blocks, self.nthread_per_block, args)

    @abstractmethod
    def pointwise_fluxes_cpu(self, q, fluxes):
        pass

    @abstractmethod
    def pointwise_fluxes_cuda(self, q, fluxes):
        pass

    @abstractmethod
    def riemann_fluxes_cpu(self, q_itf_x1: ndarray, q_itf_x2: ndarray, q_itf_x3: ndarray,
                           fluxes_itf_x1: ndarray, flux_itf_x2: ndarray, fluxes_itf_x3: ndarray) -> ndarray:
        pass

    @abstractmethod
    def riemann_fluxes_cuda(self, q_itf_x1: ndarray, q_itf_x2: ndarray, q_itf_x3: ndarray,
                            fluxes_itf_x1: ndarray, flux_itf_x2: ndarray, fluxes_itf_x3: ndarray) -> ndarray:
        pass

    @abstractmethod
    def forcing_terms(self, rhs, q):
        pass

    def _setup_cuda_indices(self):

        if self.config.num_dim == 2:
            nbsolpts = self.config.nbsolpts

            if isinstance(self.geom, Cartesian2D):
                nb_elements_x = self.config.nb_elements_horizontal
                nb_elements_z = self.config.nb_elements_vertical 
            else:
                nb_elements_x = self.config.nb_elements_horizontal
                nb_elements_z = self.config.nb_elements_horizontal

            # Create list of indices for the riemann solver
            indxl, indxr = [], []
            indzl, indzr = [], []

            ix, iz = 0, 0
            cols = 2*nbsolpts
            nb_elements = nb_elements_x * nb_elements_z
            for i in range(nb_elements_z):
                for j in range(nb_elements_x):

                    ixr = ix + 1
                    izr = iz + nb_elements_x

                    for k in range(nbsolpts):
                        ind = k + nbsolpts

                        if j + 1 < nb_elements_x:
                            indxl.append(ix * cols + ind)
                            indxr.append(ixr * cols + k)

                        if izr < nb_elements:

                            indzl.append(iz * cols + ind)
                            indzr.append(izr * cols + k)

                    # Increase the counters for each direction
                    ix += 1
                    iz += 1

            # Get boundary indices
            indbz, indbx = [], []
            count = nb_elements_x*(nb_elements_z-1)

            # Bottom boundary
            for j in range(nb_elements_x):
                for k in range(nbsolpts):
                    indbz.append(j*cols + k)

            # Top boundary
            for j in range(nb_elements_x):
                for k in range(nbsolpts):
                    indbz.append((j+count)*cols + nbsolpts + k)

            # Left boundary
            count = 0
            for j in range(nb_elements_z):
                countr = count + nb_elements_x - 1
                for k in range(nbsolpts):
                    indbx.append(count*cols + k)
                count += nb_elements_x

            # Right boundary
            count = 0
            for j in range(nb_elements_z):
                countr = count + nb_elements_x - 1
                for k in range(nbsolpts):
                    indbx.append(countr*cols + nbsolpts + k)
                count += nb_elements_x

            xp = self.device.xp
            dtype = xp.int32

            # Make these indices available in the GPU
            self.indxl = xp.array(indxl, dtype=dtype).ravel()
            self.indxr = xp.array(indxr, dtype=dtype).ravel()
            self.indyl = xp.array([], dtype=dtype).ravel()
            self.indyr = xp.array([], dtype=dtype).ravel()
            self.indzl = xp.array(indzl, dtype=dtype).ravel()
            self.indzr = xp.array(indzr, dtype=dtype).ravel()
            self.indbx = xp.array(indbx, dtype=dtype).ravel()
            self.indby = xp.array([], dtype=dtype).ravel()
            self.indbz = xp.array(indbz, dtype=dtype).ravel()
