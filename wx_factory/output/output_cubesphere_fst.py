from mpi4py import MPI

from common import Configuration
from device import Device
from geometry import CubedSphere, Metric2D, Metric3DTopo, DFROperators
from wx_mpi import ProcessTopology

from .output_cubesphere import OutputCubesphere


class OutputCubesphereFst(OutputCubesphere):
    def __init__(
        self,
        config: Configuration,
        geometry: CubedSphere,
        operators: DFROperators,
        device: Device,
        metric: Metric2D | Metric3DTopo,
        topography,
        process_topology: ProcessTopology,
    ):
        super().__init__(config, geometry, operators, device, metric, topography, process_topology)

        import rmn

        self.rank = MPI.COMM_WORLD.rank
        self.filename = f"{self.output_dir}/{self.param.base_output_file}.fst"
        self.file = None
        if self.rank == 0:
            self.file = rmn.fst24_file(self.filename, "RSF+R/W")

    def __finalize__(self):
        if self.file is not None:
            self.file.close()
