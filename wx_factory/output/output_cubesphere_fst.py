import math
import struct
import sys
from typing import Optional

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from common import Configuration, angle24
from common.definitions import (
    idx_h,
    idx_hu1,
    idx_hu2,
)
from device import Device
from geometry import CubedSphere, CubedSphere2D, CubedSphere3D, Metric2D, Metric3DTopo, DFROperators
from process_topology import ProcessTopology
from wx_mpi import SingleProcess, Conditional

from .output_cubesphere import OutputCubesphere

try:
    import rmn

    rmn_available = True
except ModuleNotFoundError:
    rmn_available = False


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

        if config.output_freq <= 0:
            return

        if not rmn_available:
            raise ValueError(f"Could not import rmn, can't use FST output manager")

        import georef

        # TODO compute proper IGs
        self.ig1 = angle24.encode(geometry.lambda0)
        self.ig2 = angle24.encode(geometry.phi0)
        self.ig3 = angle24.encode(geometry.alpha0)
        self.ig4 = _make_ig4(geometry.num_elem_horizontal, geometry.num_solpts)

        self.rank = self.comm.rank
        self.filename = f"{self.output_dir}/{self.config.base_output_file}.fst"
        self.file = None
        self.georef = None

        to_host = self.device.to_host

        # print(f"gathering lon/lat", flush=True)
        # lon = self._get_writable(self.geometry.block_lon * 180 / math.pi, num_dim=2)
        # lat = self._get_writable(self.geometry.block_lat * 180 / math.pi, num_dim=2)
        lon = self._get_writable(self.geometry.block_lon, num_dim=2)
        lat = self._get_writable(self.geometry.block_lat, num_dim=2)

        sys.stdout.flush()

        with SingleProcess() as s, Conditional(s):
            self.file = rmn.fst24_file(self.filename, "RSF+R/W")

            # print(f"lon = \n{lon / 180.0 * math.pi}")
            # print(f"lat = \n{lat}")

            for r in self.file:
                print(f"record: {r}")

            _, nj, ni = lon.shape[:3]
            self.ni = ni
            self.nj = nj * 6
            self.nk = 1  # TODO set proper nk
            print(f" nijk: {self.ni}, {self.nj}, {self.nk}")
            # If we pass the file when creating the georef, it will read the axes from it (if available)
            self.georef = georef.TGeoRef(self.ni, self.nj, "C", self.ig1, self.ig2, self.ig3, self.ig4, file=self.file)
            # self.georef.define_axes(lon, lat)
            self.georef.write("my_grid", self.file)

    def _get_writable(self, a, num_dim):
        return self.device.to_host(self._gather_field(a, num_dim))

    def _make_record(self, name, step_id, data):
        return rmn.fst_record(
            data_bits=64,
            pack_bits=64,
            data_type=rmn.FstDataType.FST_TYPE_REAL,
            data=data,
            dateo=0,
            datev=0,
            deet=int(self.config.dt),
            npas=step_id,
            ni=self.ni,
            nj=self.nj,
            nk=self.nk,
            ip1=1,
            ip2=2,
            ip3=3,
            ig1=self.ig1,
            ig2=self.ig2,
            ig3=self.ig3,
            ig4=self.ig4,
            nomvar=name[:4],
            typvar="A",
            grtyp="C",
        )

    def __write_result__(self, Q, step_id):

        def get_field(f):
            block = self.geometry.to_single_block(f)
            return self._get_writable(block, num_dim=self.num_dim)

        if isinstance(self.geometry, CubedSphere2D):
            h = get_field(Q[idx_h, ...])
            u1 = get_field(Q[idx_hu1, ...] / Q[idx_h, ...])
            u2 = get_field(Q[idx_hu2, ...] / Q[idx_h, ...])

            with SingleProcess() as s, Conditional(s):
                h_rec = self._make_record("h", step_id, h)
                self.file.write(h_rec, rewrite=0)

        else:
            raise ValueError(f"Unknown grid type {type(self.geometry)}")

    def __finalize__(self):
        if self.file is not None:
            self.file.close()
