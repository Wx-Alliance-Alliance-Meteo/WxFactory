from ..common.matmul import kron
import numpy as np
import xarray as xr
from mpi4py import MPI
import os
import shutil
import zarr
from numpy.typing import NDArray


from .output_cubesphere import OutputCubesphere
from .diagnostic import potential_vorticity, relative_vorticity
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
from ..geometry import CubedSphere2D


class OutputCubesphereZarr(OutputCubesphere):

    def __init__(
        self,
        config,
        geometry,
        operators,
        device,
        metric,
        topo,
        process_topo,
    ):
        super().__init__(config, geometry, operators, device, metric, topo, process_topo)
        if self.config.equations == "shallow_water":
            self.equ = ["h", "U", "V", "RV", "PV"]
        elif self.config.equations == "euler":
            self.equ = ["U", "V", "W", "rho", "theta", "P"]
        self.panel_id = self.process_topology.my_panel
        if config.time_start:
            self.start_time = np.datetime64(str(config.time_start).replace("t", "T"))
        else:
            self.start_time = np.datetime64("1800-01-01T00:00:00")
        self.dt = config.dt
        self.current_time_index = 0
        self.geometry = geometry
        if isinstance(self.geometry, CubedSphere2D):
            if self.geometry.z_levels:
                self.nz = len(self.geometry.z_levels)
            else:
                self.nz = 1
        else:
            self.nz = self.geometry.nk

        self.npe = 6
        self.ny = self.geometry.block_lat.shape[-2]
        self.nx = self.geometry.block_lon.shape[-1]

        # --- set file name ---
        self.filename = f"{self.output_dir}/{config.base_output_file}.zarr"
        self.marker_path = None

        # --- init ---
        if self.rank == 0:
            exists = os.path.exists(self.filename)
            if exists:
                print(f"Removing existing Zarr store: {self.filename}")
                shutil.rmtree(self.filename)
            self._output_init(self.filename)
        self.comm.Barrier()
        if self.config.equations == "euler":
            self.write_static_euler_fields()
        self.comm.Barrier()

    # --------------------------------------------------
    def _output_init(self, filename):
        # gather coordinates

        if self.nz != 1:
            ds = xr.Dataset(
                coords={
                    "time": np.array([], dtype="datetime64[ns]"),
                    "npe": np.arange(self.npe),
                    "Zdim": np.arange(self.nz),
                    "Ydim": np.arange(self.ny),
                    "Xdim": np.arange(self.nx),
                }
            )
            for var in self.equ:
                ds[var] = (
                    ("time", "npe", "Zdim", "Ydim", "Xdim"),
                    np.zeros((0, self.npe, self.nz, self.ny, self.nx), dtype=np.float64),
                )

            ds["lats"] = (("npe", "Ydim", "Xdim"), np.zeros((self.npe, self.ny, self.nx), dtype=np.float64))

            ds["lons"] = (("npe", "Ydim", "Xdim"), np.zeros((self.npe, self.ny, self.nx), dtype=np.float64))

            # initialize empty variable array
            if self.config.equations == "euler":

                ds["elev"] = (
                    ("npe", "Zdim", "Ydim", "Xdim"),
                    np.zeros((self.npe, self.nz, self.ny, self.nx), dtype=np.float64),
                )

                ds["topo"] = (("npe", "Ydim", "Xdim"), np.zeros((self.npe, self.ny, self.nx), dtype=np.float64))

                ds["volume"] = (
                    ("npe", "Zdim", "Ydim", "Xdim"),
                    np.zeros((self.npe, self.nz, self.ny, self.nx), dtype=np.float64),
                )

            # chunking
            ds = ds.chunk(
                {
                    "time": 1,
                    "npe": 1,
                    "Zdim": self.nz,
                    "Ydim": self.ny,
                    "Xdim": self.nx,
                }
            )
        else:
            ds = xr.Dataset(
                coords={
                    "time": np.array([], dtype="datetime64[ns]"),
                    "npe": np.arange(self.npe),
                    "Ydim": np.arange(self.ny),
                    "Xdim": np.arange(self.nx),
                }
            )
            for var in self.equ:
                ds[var] = (("time", "npe", "Ydim", "Xdim"), np.zeros((0, self.npe, self.ny, self.nx), dtype=np.float64))

            # chunking
            ds = ds.chunk(
                {
                    "time": 1,
                    "npe": 1,
                    "Ydim": self.ny,
                    "Xdim": self.nx,
                }
            )

        if self.rank == 0:
            ds["time"].encoding = {
                "units": "seconds since 1800-01-01 00:00:00",
            }
            ds.to_zarr(filename, mode="w")

    # --------------------------------------------------
    def __write_result__(self, Q, step_id):
        time_val = self.start_time + np.timedelta64(int(step_id * self.dt), "s")
        # nx = self.geometry.block_lat.shape[-1]

        t_index = self._append_time_step(
            time_val,
            self.nz,
            self.ny,
            self.nx,
        )

        if self.config.equations == "shallow_water":
            if self.nz > 1:
                for k in range(self.nz):
                    h = Q[idx_h, k, ...]

                    if self.topo is not None:
                        h = h + self.topo.hsurf

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(h), "h", t_index, k)

                    u1 = Q[idx_hu1, k, ...] / h
                    u2 = Q[idx_hu2, k, ...] / h

                    u, v = self.geometry.contra2wind(u1, u2)

                    rv = relative_vorticity(u1, u2, self.metric, self.operators)

                    pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(u), "U", t_index, k)

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(v), "V", t_index, k)

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(rv), "RV", t_index, k)

                    self.store_variable_zarr_Zdim(self.geometry.to_single_block(pv), "PV", t_index, k)

            else:

                h = Q[idx_h, :, :]

                if self.topo is not None:
                    h = h + self.topo.hsurf

                self.store_variable_zarr(self.geometry.to_single_block(h), "h", t_index)

                u1 = Q[idx_hu1, :, :] / h
                u2 = Q[idx_hu2, :, :] / h

                u, v = self.geometry.contra2wind(u1, u2)

                rv = relative_vorticity(u1, u2, self.metric, self.operators)

                pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                self.store_variable_zarr(self.geometry.to_single_block(u), "U", t_index)

                self.store_variable_zarr(self.geometry.to_single_block(v), "V", t_index)

                self.store_variable_zarr(self.geometry.to_single_block(rv), "RV", t_index)

                self.store_variable_zarr(self.geometry.to_single_block(pv), "PV", t_index)

            return

        elif self.config.equations == "euler":

            rho = Q[idx_rho, ...]
            u1 = Q[idx_rho_u1, ...] / rho
            u2 = Q[idx_rho_u2, ...] / rho
            u3 = Q[idx_rho_u3, ...] / rho
            theta = Q[idx_rho_theta, ...] / rho

            u, v, w = self.geometry.contra2wind_3d(u1, u2, u3, self.metric)

            p = p0 * (Q[idx_rho_theta] * Rd / p0) ** (cpd / cvd)

            for k in range(self.nz):

                self.store_variable_zarr_Zdim(self.geometry.to_single_block(u), "U", t_index, k)

                self.store_variable_zarr_Zdim(self.geometry.to_single_block(v), "V", t_index, k)

                self.store_variable_zarr_Zdim(self.geometry.to_single_block(w), "W", t_index, k)

                self.store_variable_zarr_Zdim(self.geometry.to_single_block(rho), "rho", t_index, k)

                self.store_variable_zarr_Zdim(self.geometry.to_single_block(theta), "theta", t_index, k)

                self.store_variable_zarr_Zdim(self.geometry.to_single_block(p), "P", t_index, k)

            return

    # --------------------------------------------------
    def __finalize__(self):
        return

    def _append_time_step(self, time_val, nz, ny, nx):
        variables = {}
        if self.rank == 0:
            if nz != 1:
                for var in self.equ:
                    variables[var] = (
                        ("time", "npe", "Zdim", "Ydim", "Xdim"),
                        np.full((1, self.npe, self.nz, self.ny, self.nx), np.nan, dtype=np.float64),
                    )
                    dummy = xr.Dataset(
                        variables,
                        coords={
                            "time": np.array([time_val], dtype="datetime64[s]"),
                            "npe": np.arange(self.npe),
                            "Zdim": np.arange(nz),
                            "Ydim": np.arange(ny),
                            "Xdim": np.arange(nx),
                        },
                    )
            else:
                for var in self.equ:
                    variables[var] = (
                        ("time", "npe", "Ydim", "Xdim"),
                        np.full((1, self.npe, self.ny, self.nx), np.nan, dtype=np.float64),
                    )
                    dummy = xr.Dataset(
                        variables,
                        coords={
                            "time": np.array([time_val], dtype="datetime64[s]"),
                            "npe": np.arange(self.npe),
                            "Ydim": np.arange(ny),
                            "Xdim": np.arange(nx),
                        },
                    )

            dummy["time"].encoding = {
                "units": "seconds since 1800-01-01 00:00:00",
            }

            dummy.to_zarr(
                self.filename,
                append_dim="time",
            )

        if self.rank == 0:
            t_index = self.current_time_index
            self.current_time_index += 1
        else:
            t_index = None

        t_index = self.comm.bcast(t_index, root=0)
        self.comm.Barrier()

        return t_index

    def store_variable_zarr(self, field, variable_name, t_index):

        fields = self._gather_field(field, self.num_dim)

        if fields is None:
            return

        if self.rank == 0:

            root = zarr.open_group(
                self.filename,
                mode="r+",
            )

            for face_idx, face in enumerate(fields):
                root[variable_name][t_index, face_idx, :, :] = self.device.to_host(face)

    def store_variable_zarr_Zdim(self, field, variable_name, t_index, level_idx):

        fields = self._gather_field(field, self.num_dim)

        if fields is None:
            return

        if self.rank == 0:
            root = zarr.open_group(
                self.filename,
                mode="r+",
            )

            for face_idx, face in enumerate(fields):
                if self.config.equations == "shallow_water":
                    root[variable_name][t_index, face_idx, level_idx, :, :] = self.device.to_host(face)

                elif self.config.equations == "euler":
                    root[variable_name][t_index, face_idx, level_idx, :, :] = self.device.to_host(face[level_idx, :, :])

    def write_static_euler_fields(self):

        lats = self._gather_field(self.geometry.block_lat * 180 / np.pi, 2)

        lons = self._gather_field(self.geometry.block_lon * 180 / np.pi, 2)

        elev = self._gather_field(self.geometry.coordVec_latlon[2, :, :, :], 3)

        topo = self._gather_field(self.geometry.zbot, 2)

        volume = self._gather_field(self.geometry.to_single_block(self._cell_volume()), 3)

        if self.rank != 0:
            return

        root = zarr.open_group(
            self.filename,
            mode="r+",
        )

        root["lats"][:] = self.device.to_host(lats)
        root["lons"][:] = self.device.to_host(lons)
        root["elev"][:] = self.device.to_host(elev)
        root["topo"][:] = self.device.to_host(topo)
        root["volume"][:] = self.device.to_host(volume)

    def _cell_volume(self) -> NDArray:
        """Volume associated with each solution point.

        On the cubed sphere the elements are uniform in the computational coordinates, so the volume
        of a solution point is sqrt(G) times its tensor-product Gauss-Legendre quadrature weight,
        times the (constant) volume of a reference element. The solution points inside an element are
        ordered with x1 varying fastest, then x2, then x3."""
        geom = self.geometry
        w = geom.glweights
        w3d = kron(w, kron(w, w))  # ordering: x3 slowest, x1 fastest

        elem_volume = geom.delta_x1 * geom.delta_x2 * geom.delta_x3 / 8.0
        return self.metric.sqrtG_new * w3d * elem_volume
