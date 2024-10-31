from common.device import Device
from .geometry import Geometry


class CubedSphere(Geometry):
    def __init__(self, nbsolpts: int, device: Device, verbose: bool | None = False) -> None:
        super().__init__(nbsolpts, device, verbose)
