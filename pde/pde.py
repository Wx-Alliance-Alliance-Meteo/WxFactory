from abc import ABC, abstractmethod
from numpy import ndarray


def get_pde(name):
    if name == "euler-cartesian":
        from pde.pde_euler_cartesian import PDEEulerCartesian
        return PDEEulerCartesian

    elif name == "euler-cubesphere":
        from pde.pde_euler_cubedsphere import PDEEulerCubedSphere
        return PDEEulerCubedSphere


class PDE(ABC):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.num_dim = config.num_dim

        # Set the CUDA indices to None
        if self.config.device == 'cuda':
            self.indxl = None
            self.indxr = None
            self.indyl = None
            self.indyr = None
            self.indzl = None
            self.indzr = None
            self.indbx = None
            self.indby = None
            self.indbz = None

            # Written for only 2D for now
            self.get_riemann_indices(config.nb_elements_horizontal,
                                     config.nb_elements_vertical,
                                     config.nbsolpts)


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

    def get_riemann_indices(self, nb_elements_x, nb_elements_z, nbsolpts):

        if self.config.num_dim == 2:
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
            self.indzl = xp.array(indzl, dtype=dtype).ravel()
            self.indzr = xp.array(indzr, dtype=dtype).ravel()
            self.indbx = xp.array(indbx, dtype=dtype).ravel()
            self.indbz = xp.array(indbz, dtype=dtype).ravel()
