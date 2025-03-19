from device import Device
from .geometry import Geometry


class CubedSphere(Geometry):
    def __init__(
        self,
        num_elem_horizontal: int,
        num_elem_vertical: int,
        num_solpts: int,
        lambda0: float,
        phi0: float,
        alpha0: float,
        device: Device,
        verbose: bool | None = False,
    ) -> None:
        super().__init__(num_solpts, device, verbose)
        self.num_elem_horizontal = num_elem_horizontal
        self.num_elem_vertical = num_elem_vertical
        self.lambda0 = lambda0
        self.phi0 = phi0
        self.alpha0 = alpha0
