from common.device import Device
from .geometry import Geometry


class CubedSphere(Geometry):
    def __init__(self, num_solpts: int, device: Device, verbose: bool | None = False) -> None:
        super().__init__(num_solpts, device, verbose)
