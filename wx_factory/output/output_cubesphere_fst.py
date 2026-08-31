import math
import numpy as np
from ..common import Configuration, angle24
from ..common.definitions import (
    idx_h,
    idx_hu1,
    idx_hu2,
    idx_hu2,
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
    idx_rho_theta,
    cpd,
    cvd,
    p0,
    Rd,
)
from ..context import Context
from ..geometry import CubedSphere, CubedSphere2D, DFROperators, Metric2D, Metric3DTopo
from ..process_topology import ProcessTopology
from ..wx_mpi import Conditional, SingleProcess
from .output_cubesphere import OutputCubesphere
from .diagnostic import potential_vorticity, relative_vorticity
import torch
import os

try:
    import rmn

    rmn_available = True
except ModuleNotFoundError:
    rmn_available = False
    raise


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

        # import georef

        # TODO compute proper IGs
        self.ig1 = angle24.encode(geometry.lambda0)
        self.ig2 = angle24.encode(geometry.phi0)
        self.ig3 = angle24.encode(geometry.alpha0)
        # self.ig4 = georef.cubed_sphere.encodeig4(geometry.num_elements_horizontal, geometry.num_solpts)
        self.dtype = torch.finfo(self.context.real_dtype).bits
        self.height = 1

        self.filename = f"{self.output_dir}/{self.config.base_output_file}.fst"
        self.file: rmn.fst24_file = None
        self.georef = None

        with SingleProcess(self.comm) as s, Conditional(s):
            if os.path.exists(self.filename):
                print(f"Removing existing fst: {self.filename}")
                os.remove(self.filename)

        if config.time_start:
            self.start_time = np.datetime64(str(config.time_start).replace("t", "T"))
        else:
            self.start_time = np.datetime64("1980-01-01T00:00:00")
        self.actual_time = np.datetime64("1980-01-01T00:00:00")
        if isinstance(self.geometry, CubedSphere2D):
            if len(self.geometry.z_levels) > 1:
                self.nk = len(self.geometry.z_levels) - 1
            else:
                self.nk = 1
        else:
            self.nk = self.geometry.nk

        to_host = lambda a: self.context.to_host(a) if a is not None else None

        # if self.rank == 0:
        lon = to_host(self._gather_field(self.geometry.block_lon * 180 / math.pi, num_dim=2))
        lat = to_host(self._gather_field(self.geometry.block_lat * 180 / math.pi, num_dim=2))

        with SingleProcess() as s, Conditional(s):
            self.file = rmn.fst24_file(self.filename, "RSF+R/W")

            for r in self.file:
                print(f"record: {r}")

            _, nj, ni = lon.shape[:3]
            self.ni = ni
            self.nj = nj * 6
            print(f" nijk: {self.ni}, {self.nj}, {self.nk}", flush=True)

            # If we pass the file when creating the georef, it will read the axes from it (if available)
            # self.georef = georef.GeoRef(self.ni, self.nj, "Q", self.ig1, self.ig2, self.ig3, self.ig4, self.file)
            # self.georef.write_fst(self.file, self.ig1, self.ig2, self.ig3, self.ig4, "my_grid")

    def _make_record(self, name, step_id, data):
        return rmn.fst_record(
            data_bits=self.dtype,
            pack_bits=self.dtype,
            data_type=rmn.FstDataType.FST_TYPE_REAL_TURBOPACK,
            data=data,
            dateo=rmn.fst_date.encode_date(self.start_time),
            datev=rmn.fst_date.encode_date(self.actual_time),
            deet=int(self.config.dt),
            npas=step_id,
            ni=self.ni,
            nj=self.nj,
            nk=self.nk,
            ip1=1,
            ip2=2,
            ip3=self.height,
            ig1=self.ig1,
            ig2=self.ig2,
            ig3=self.ig3,
            ig4=10,
            nomvar=name[:4],
            typvar="A",
            grtyp="Q",
        )

    def __write_result__(self, Q, step_id):
        self.actual_time = self.start_time + np.timedelta64(int(step_id * self.config.dt), "s")
        if self.config.equations == "shallow_water":
            for k in range(self.nk):
                if self.nk > 1:
                    h = Q[idx_h, k, ...]
                else:
                    h = Q[idx_h, :, :]

                if self.topo is not None:
                    h = h + self.topo.hsurf
                self.height = k + 1

                self.store_variable(self.geometry.to_single_block(h), "h", step_id)

                if self.nk > 1:
                    u1 = Q[idx_hu1, k, ...] / h
                    u2 = Q[idx_hu2, k, ...] / h
                else:
                    u1 = Q[idx_hu1, :, :] / h
                    u2 = Q[idx_hu2, :, :] / h

                u, v = self.geometry.contra2wind(u1, u2)

                rv = relative_vorticity(u1, u2, self.metric, self.operators)

                pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                self.store_variable(self.geometry.to_single_block(u), "U", step_id)

                self.store_variable(self.geometry.to_single_block(v), "V", step_id)

                self.store_variable(self.geometry.to_single_block(rv), "RV", step_id)

                self.store_variable(self.geometry.to_single_block(pv), "PV", step_id)

        elif self.config.equations == "euler":

            rho = Q[idx_rho, ...]
            u1 = Q[idx_rho_u1, ...] / rho
            u2 = Q[idx_rho_u2, ...] / rho
            u3 = Q[idx_rho_u3, ...] / rho
            theta = Q[idx_rho_theta, ...] / rho

            u, v, w = self.geometry.contra2wind_3d(u1, u2, u3, self.metric)

            p = p0 * (Q[idx_rho_theta] * Rd / p0) ** (cpd / cvd)

            self.store_variable(self.geometry.to_single_block(u), "U", step_id)

            self.store_variable(self.geometry.to_single_block(v), "V", step_id)

            self.store_variable(self.geometry.to_single_block(w), "W", step_id)

            self.store_variable(self.geometry.to_single_block(rho), "rho", step_id)

            self.store_variable(self.geometry.to_single_block(theta), "theta", step_id)

            self.store_variable(self.geometry.to_single_block(p), "P", step_id)

            return

    def store_variable(self, field, variable_name, step_id):

        fields = self._gather_field(field, self.num_dim)

        if fields is None:
            return

        with SingleProcess(self.comm) as s, Conditional(s):
            self.file.write(self._make_record(variable_name, step_id, self.context.to_host(fields)), rewrite=0)

    def __finalize__(self):
        if self.file is not None:
            self.file.close()
