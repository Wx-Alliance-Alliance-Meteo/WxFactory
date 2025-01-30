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
from wx_mpi import ProcessTopology, SingleProcess, Conditional

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

        print(f"phi0 = {geometry.phi0}")

        # TODO compute proper IGs
        self.ig1 = angle24.encode(geometry.lambda0)
        self.ig2 = angle24.encode(geometry.phi0)
        self.ig3 = angle24.encode(geometry.alpha0)
        self.ig4 = geometry.num_solpts

        self.rank = MPI.COMM_WORLD.rank
        self.filename = f"{self.output_dir}/{self.config.base_output_file}.fst"
        self.file = None
        self.georef = None

        to_host = self.device.to_host

        self.is_3d = isinstance(self.geometry, CubedSphere3D)

        # print(f"gathering lon/lat", flush=True)
        lon = to_host(self._gather_field(self.geometry.block_lon * 180 / math.pi))
        lat = to_host(self._gather_field(self.geometry.block_lat * 180 / math.pi))

        sys.stdout.flush()

        MPI.COMM_WORLD.barrier()
        print(
            f"{MPI.COMM_WORLD.rank}: {geometry.lon_p:9.6f}, {geometry.lat_p:9.6f}, {geometry.angle_p:9.6f}", flush=True
        )
        MPI.COMM_WORLD.barrier()

        with SingleProcess() as s, Conditional(s):
            self.file = rmn.fst24_file(self.filename, "RSF+R/W")
            # for i in range(0xFFFFFF):
            #     j = angle24.encode(angle24.decode(i))
            #     if i != j:
            #         print(
            #             f"Difference for {i} (0x{i:x}, {angle24.decode(i)}), "
            #             f"reencodes to {j} (0x{j:x}, {angle24.decode(j)})",
            #             flush=True,
            #         )
            #         raise ValueError
            #     if i % 100000 == 0:
            #         print(f"i = {i}", flush=True)

            print(
                f"pi/2:   0x{angle24.encode(math.pi/2):x} ({angle24.decode(0x0) + math.pi/2})\n"
                f"3pi/8:  0x{angle24.encode(3*math.pi/8):x} ({angle24.decode(0xe00000) - 3*math.pi/8})\n"
                f"pi/4:   0x{angle24.encode(math.pi/4):x} ({angle24.decode(0xc00000) - math.pi/4})\n"
                f"pi/8:   0x{angle24.encode(math.pi/8):x} ({angle24.decode(0xa00000) - math.pi/8})\n"
                f"0:      0x{angle24.encode(0.):x} ({angle24.decode(0x800000)})\n"
                f"-pi/8:  0x{angle24.encode(-math.pi/8):x} ({angle24.decode(0x600000) + math.pi/8})\n"
                f"-pi/4:  0x{angle24.encode(-math.pi/4):x} ({angle24.decode(0x400000) + math.pi/4})\n"
                f"-3pi/8: 0x{angle24.encode(-3*math.pi/8):x} ({angle24.decode(0x200000) + 3*math.pi/8})\n"
                f"-pi/2:  0x{angle24.encode(-math.pi/2):x} ({angle24.decode(0x0) + math.pi/2})\n"
            )

            self.nk, self.nj, self.ni = lon.shape[:3]
            # If we pass the file when creating the georef, it will read the axes from it (if available)
            self.georef = georef.TGeoRef(self.ni, self.nj, "C", self.ig1, self.ig2, self.ig3, self.ig4, file=None)
            self.georef.define_axes(lon, lat)
            self.georef.write("my_grid", self.file)
            # # print(f"lon shape {lon.shape}, lat shape {lat.shape}")
            # # print(f"{lon[0, 0, :3]}, {lat[0, 0, :3]}")
            # rec = rmn.fst_record(
            #     data_bits=64,
            #     pack_bits=64,
            #     data_type=rmn.FstDataType.FST_TYPE_REAL,
            #     data=to_host(lon),
            #     dateo=0,
            #     datev=0,
            #     deet=0,
            #     npas=0,
            #     ni=ni,
            #     nj=nj,
            #     nk=nk,
            #     ip1=1,
            #     ip2=2,
            #     ip3=3,
            #     ig1=1,
            #     ig2=2,
            #     ig3=3,
            #     ig4=4,
            #     nomvar=">>",
            #     typvar="Y",
            #     grtyp="C",
            # )
            # self.file.write(rec, 2)

    def _gather_field(self, field: NDArray) -> Optional[NDArray]:
        xp = self.device.xp
        field_list = super()._gather_field(field)

        if field_list is None:
            return None

        if self.is_3d:
            return xp.concatenate(field_list, axis=1)
        else:
            fields = xp.concatenate(field_list, axis=0)
            return xp.expand_dims(fields, axis=0)

    def _make_record(self, name, step_id, data):
        return rmn.fst_record(
            data_bits=64,
            pack_bits=64,
            data_type=rmn.FstDataType.FST_TYPE_REAL,
            data=data,
            dateo=0,
            datev=0,
            deet=int(self.param.dt),
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
            to_block = self.geometry.to_single_block
            gather = self._gather_field
            to_host = self.device.to_host
            return to_host(gather(to_block(f)))

        if isinstance(self.geometry, CubedSphere2D):
            h = get_field(Q[idx_h, ...])
            u1 = get_field(Q[idx_hu1, ...] / Q[idx_h, ...])
            u2 = get_field(Q[idx_hu2, ...] / Q[idx_h, ...])

            with SingleProcess() as s, Conditional(s):
                h_rec = self._make_record("h", step_id, h)
                self.file.write(h_rec)

        else:
            raise ValueError(f"Unknown grid type {type(self.geometry)}")

    def __finalize__(self):
        if self.file is not None:
            self.file.close()
