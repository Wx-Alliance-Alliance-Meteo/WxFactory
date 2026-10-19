import os
import shutil

import numpy as np
import zarr
from numpy.typing import NDArray

from ..common.definitions import (
    Rd,
    cpd,
    cvd,
    idx_h,
    idx_hu1,
    idx_hu2,
    idx_rho,
    idx_rho_theta,
    p0,
)
from ..common.euler_state import physical_winds
from ..common.matmul import kron
from ..geometry import CubedSphere2D
from .diagnostic import potential_vorticity, relative_vorticity
from .output_cubesphere import OutputCubesphere


class OutputCubesphereZarr(OutputCubesphere):
    def __init__(
        self,
        config,
        geometry,
        operators,
        context,
        metric,
        topo,
        process_topo,
    ):
        super().__init__(config, geometry, operators, context, metric, topo, process_topo)
        if self.config.equations == "shallow_water":
            self.equ = ["h", "U", "V", "RV", "PV"]
        elif self.config.equations == "euler":
            self.equ = ["U", "V", "W", "rho", "theta", "P"]
            if config.case_number in (11, 12, 13):
                count = 1 if config.case_number == 12 else 4
                self.equ += [f"q{index}" for index in range(1, count + 1)]
        self.panel_id = self.process_topology.my_panel
        if config.time_start:
            self.start_time = np.datetime64(str(config.time_start).replace("t", "T"))
        else:
            self.start_time = np.datetime64("1800-01-01T00:00:00")
        self.dt = config.dt
        self.num_steps = int(np.ceil(config.t_end / config.dt) + 1)
        self.current_time_index = 0
        self.geometry = geometry
        if isinstance(self.geometry, CubedSphere2D):
            if len(self.geometry.z_levels) > 1:
                self.nz = len(self.geometry.z_levels) - 1
            else:
                self.nz = 1
        else:
            self.nz = self.geometry.nk

        self.nfaces = 6
        self.ny = self.geometry.block_lat.shape[-2]
        self.nx = self.geometry.block_lon.shape[-1]

        self.panel_x = self.context.to_host(self._gather_panel(self.geometry.x1[...]))
        self.panel_y = self.context.to_host(self._gather_panel(self.geometry.x2[...]))
        if self.nz != 1 and self.config.equations == "euler":
            self.zcoord = self.context.to_host(self.geometry.x3[:, 0, 0])
        else:
            self.zcoord = np.arange(self.nz)

        self.filename = f"{self.output_dir}/{config.base_output_file}.zarr"
        self.marker_path = None

        if self.rank == 0:
            exists = os.path.exists(self.filename)
            if exists:
                print(f"Removing existing Zarr store: {self.filename}")
                shutil.rmtree(self.filename)
            self._output_init(self.filename)
            self.ds_zarr = zarr.open_group(self.filename, mode="r+")
        self.write_static_fields()

    def _output_init(self, filename):
        times = np.array(
            [self.start_time + np.timedelta64(int(i * self.dt), "s") for i in range(self.num_steps)],
            dtype="datetime64[ns]",
        )
        ds_zarr = zarr.open_group(filename, mode="w")
        if self.nz != 1:
            for var in self.equ:
                ds_zarr.create_array(
                    var,
                    shape=(self.num_steps, self.nfaces, self.nz, self.ny, self.nx),
                    chunks=(1, 1, self.nz, self.ny, self.nx),
                    dtype=np.float64,
                )

            ds_zarr.create_array(
                "Zdim",
                data=self.zcoord,
            )

        else:
            for var in self.equ:
                ds_zarr.create_array(
                    var,
                    shape=(self.num_steps, self.nfaces, self.ny, self.nx),
                    chunks=(1, 1, self.ny, self.nx),
                    dtype=np.float64,
                )

        ds_zarr.create_array(
            "time",
            data=times,
        )

        ds_zarr.create_array(
            "nfaces",
            data=np.arange(self.nfaces),
        )

        ds_zarr.create_array(
            "Xdim",
            data=self.panel_x,
        )

        ds_zarr.create_array(
            "Ydim",
            data=self.panel_y,
        )

        ds_zarr.create_array(
            "lats",
            shape=(self.nfaces, self.ny, self.nx),
            chunks=(1, self.ny, self.nx),
            dtype=np.float64,
        )

        ds_zarr.create_array(
            "lons",
            shape=(self.nfaces, self.ny, self.nx),
            chunks=(1, self.ny, self.nx),
            dtype=np.float64,
        )

        if self.config.equations == "euler":
            ds_zarr.create_array(
                "elev",
                shape=(self.nfaces, self.nz, self.ny, self.nx),
                chunks=(1, self.nz, self.ny, self.nx),
                dtype=np.float64,
            )

            ds_zarr.create_array(
                "topo",
                shape=(self.nfaces, self.ny, self.nx),
                chunks=(1, self.ny, self.nx),
                dtype=np.float64,
            )

            ds_zarr.create_array(
                "volume",
                shape=(self.nfaces, self.nz, self.ny, self.nx),
                chunks=(1, self.nz, self.ny, self.nx),
                dtype=np.float64,
            )

    def __write_result__(self, Q, step_id):

        if self.config.equations == "shallow_water":
            if self.nz > 1:
                for k in range(self.nz):
                    h = Q[idx_h, k, ...]

                    if self.topo is not None:
                        h = h + self.topo.hsurf

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(h), "h", step_id, k)

                    u1 = Q[idx_hu1, k, ...] / h
                    u2 = Q[idx_hu2, k, ...] / h

                    u, v = self.geometry.contra2wind(u1, u2)

                    rv = relative_vorticity(u1, u2, self.metric, self.operators)

                    pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(u), "U", step_id, k)

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(v), "V", step_id, k)

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(rv), "RV", step_id, k)

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(pv), "PV", step_id, k)

            else:
                h = Q[idx_h, :, :]

                if self.topo is not None:
                    h = h + self.topo.hsurf

                self.store_variable_zarr(self.geometry.to_single_block(h), "h", step_id)

                u1 = Q[idx_hu1, :, :] / h
                u2 = Q[idx_hu2, :, :] / h

                u, v = self.geometry.contra2wind(u1, u2)

                rv = relative_vorticity(u1, u2, self.metric, self.operators)

                pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                self.store_variable_zarr(self.geometry.to_single_block(u), "U", step_id)

                self.store_variable_zarr(self.geometry.to_single_block(v), "V", step_id)

                self.store_variable_zarr(self.geometry.to_single_block(rv), "RV", step_id)

                self.store_variable_zarr(self.geometry.to_single_block(pv), "PV", step_id)

            return

        elif self.config.equations == "euler":
            rho = Q[idx_rho, ...]
            theta = Q[idx_rho_theta, ...] / rho

            u, v, w = physical_winds(Q, self.geometry, self.metric)

            p = p0 * (Q[idx_rho_theta] * Rd / p0) ** (cpd / cvd)

            self.store_variable_zarr(self.geometry.to_single_block(u), "U", step_id)

            self.store_variable_zarr(self.geometry.to_single_block(v), "V", step_id)

            self.store_variable_zarr(self.geometry.to_single_block(w), "W", step_id)

            self.store_variable_zarr(self.geometry.to_single_block(rho), "rho", step_id)

            self.store_variable_zarr(self.geometry.to_single_block(theta), "theta", step_id)

            self.store_variable_zarr(self.geometry.to_single_block(p), "P", step_id)
            if self.config.case_number in (11, 12, 13):
                for index in range(5, Q.shape[0]):
                    self.store_variable_zarr(self.geometry.to_single_block(Q[index] / rho), f"q{index - 4}", step_id)

            return

    def __finalize__(self):
        return

    def store_variable_zarr(self, field, variable_name, step_id):

        fields = self._gather_field(field, self.num_dim)

        if fields is None:
            return

        if self.rank == 0:
            self.ds_zarr[variable_name][step_id] = self.context.to_host(fields)

    def store_variable_zarr_Zdim(self, field, variable_name, step_id, level_idx):

        fields = self._gather_field(field, self.num_dim)

        if fields is None:
            return

        if self.rank == 0:
            self.ds_zarr[variable_name][step_id, :, level_idx, :, :] = self.context.to_host(fields)

    def write_static_fields(self):

        lats = self._gather_field(self.geometry.block_lat * 180 / np.pi, 2)

        lons = self._gather_field(self.geometry.block_lon * 180 / np.pi, 2)

        if self.config.equations == "euler":
            elev = self._gather_field(self.geometry.coordVec_latlon[2, :, :, :], 3)

            topo = self._gather_field(self.geometry.zbot, 2)

            volume = self._gather_field(self.geometry.to_single_block(self._cell_volume()), 3)

        if self.rank != 0:
            return

        self.ds_zarr["lats"][:] = self.context.to_host(lats)
        self.ds_zarr["lons"][:] = self.context.to_host(lons)
        if self.config.equations == "euler":
            self.ds_zarr["elev"][:] = self.context.to_host(elev)
            self.ds_zarr["topo"][:] = self.context.to_host(topo)
            self.ds_zarr["volume"][:] = self.context.to_host(volume)

    def _cell_volume(self) -> NDArray:
        """The reference-coordinate Jacobian already includes element scaling.

        Tensor-product quadrature weights use x1 fastest, then x2, then x3.
        """
        geom = self.geometry
        w = geom.glweights
        w3d = kron(w, kron(w, w))

        return self.metric.volume.volume_jacobian * w3d
