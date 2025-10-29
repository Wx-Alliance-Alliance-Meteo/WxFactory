from device import Device
from .geometry import Geometry
from process_topology import ProcessTopology


class CubedSphere(Geometry):
    def __init__(
        self,
        num_elem_horizontal: int,
        num_elem_vertical: int,
        num_solpts: int,
        total_num_elements_horizontal: int,
        lambda0: float,
        phi0: float,
        alpha0: float,
        process_topology: ProcessTopology,
        verbose: bool | None = False,
    ) -> None:
        super().__init__(
            num_solpts,
            num_elem_horizontal,
            num_elem_vertical,
            total_num_elements_horizontal,
            process_topology.device,
            verbose,
        )
        self.process_topology = process_topology
        self.lambda0 = lambda0
        self.phi0 = phi0
        self.alpha0 = alpha0
