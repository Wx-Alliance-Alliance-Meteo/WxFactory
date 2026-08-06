from ..common import Configuration, angle24
from ..common.definitions import (
    idx_h,
    idx_hu1,
    idx_hu2,
)
from ..context import Context
from ..geometry import CubedSphere, CubedSphere2D, DFROperators, Metric2D, Metric3DTopo
from ..process_topology import ProcessTopology
from ..wx_mpi import Conditional, SingleProcess
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
        context: Context,
        metric: Metric2D | Metric3DTopo,
        topography,
        process_topology: ProcessTopology,
    ):
        super().__init__(config, geometry, operators, context, metric, topography, process_topology)

        if config.output_freq <= 0:
            return

        if not rmn_available:
            raise ValueError("Could not import rmn, can't use FST output manager")

        import georef

        # TODO compute proper IGs
        self.ig1 = angle24.encode(geometry.lambda0)
        self.ig2 = angle24.encode(geometry.phi0)
        self.ig3 = angle24.encode(geometry.alpha0)
        self.ig4 = georef.cubed_sphere.encodeig4(geometry.num_elements_horizontal, geometry.num_solpts)

        self.rank = self.comm.rank
        self.filename = f"{self.output_dir}/{self.config.base_output_file}.fst"
        self.file: rmn.fst24_file = None
        self.georef = None

        to_host = self.context.to_host

        lon = self._get_writable(self.geometry.block_lon, num_dim=2)
        lat = self._get_writable(self.geometry.block_lat, num_dim=2)

        with SingleProcess() as s, Conditional(s):
            self.file = rmn.fst24_file(self.filename, "RSF+R/W")

            for r in self.file:
                print(f"record: {r}")

            _, nj, ni = lon.shape[:3]
            self.ni = ni
            self.nj = nj * 6
            self.nk = 1  # TODO set proper nk
            print(f" nijk: {self.ni}, {self.nj}, {self.nk}", flush=True)

            # If we pass the file when creating the georef, it will read the axes from it (if available)
            self.georef = georef.GeoRef(self.ni, self.nj, "Q", self.ig1, self.ig2, self.ig3, self.ig4, self.file)
            self.georef.write_fst(self.file, self.ig1, self.ig2, self.ig3, self.ig4, "my_grid")

    def _get_writable(self, a, num_dim):
        return self.context.to_host(self._gather_field(a, num_dim))

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
            grtyp="Q",
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
