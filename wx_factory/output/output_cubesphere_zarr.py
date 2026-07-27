import numpy as np
import xarray as xr
from mpi4py import MPI
import os
import shutil
import zarr


from .output_cubesphere import OutputCubesphere
from .diagnostic import potential_vorticity, relative_vorticity
from ..common.definitions import idx_h, idx_hu1, idx_hu2, idx_hu2, idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho_theta, cpd, cvd, p0, Rd


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
        # self.max_time = len(self.dataset.time)

        # --- get real ERA5 time ---
        # self.all_times = self.dataset.time.values
        # self.time_to_index = {str(t): i for i, t in enumerate(self.all_times)}
        self.filename = f"{self.output_dir}/{config.base_output_file}.zarr"
        self.marker_path = None

        # --- init once per year ---
        # check only on rank 0
        if self.rank == 0:
            exists = os.path.exists(self.filename)
            if exists:
                print(f"Removing existing Zarr store: {self.filename}")
                shutil.rmtree(self.filename)
            self._output_init(self.filename)
        if self.config.equations == "euler":
            self.write_static_euler_fields()
        self.comm.Barrier()

    # --------------------------------------------------
    def _output_init(self, filename):
        # gather coordinates
        lons = self.device.to_host(self.geometry.block_lon * 180 / np.pi)
        lats = self.device.to_host(self.geometry.block_lat * 180 / np.pi)

        #nz = Q.shape[1] if Q.ndim == 5 else 1
        if self.geometry.nk > 1: 
            nz = self.geometry.nk
        else:
            if self.geometry.z_levels:
                if type(self.geometry.z_levels) == int:
                    nz = self.geometry.z_levels
                else:
                    nz = len(self.geometry.z_levels)
            else:
                nz = 1
        npe = 6
        ny = lons.shape[-2]
        nx = lats.shape[-1]

        if nz != 1:
            if self.config.equations == "shallow_water":
                ds = xr.Dataset(
                    coords={
                        "time": np.array([], dtype="datetime64[ns]"),
                        "equations": np.array(self.equ, dtype=object),
                        "z": np.arange(nz),
                        "faces": np.arange(npe),
                        "y": np.arange(ny),
                        "x": np.arange(nx),
                    }
                )

                # initialize empty variable array
                shape = (0, len(self.equ), nz, npe, ny, nx)
            
                ds["data"] = (("time", "equations", "z", "faces", "y", "x"), np.zeros(shape, dtype=np.float64))

                # chunking
                ds = ds.chunk(
                    {
                        "time": 1,
                        "equations": 1,
                        "z": nz,
                        "faces": 1,
                        "y": ny,
                        "x": nx,
                    }
                )

            elif self.config.equations == "euler":
                ds = xr.Dataset(
                    coords={
                        "time": np.array([], dtype="datetime64[ns]"),
                        "equations": np.array(self.equ, dtype=object),
                        "faces": np.arange(npe),
                        "z": np.arange(nz),
                        "y": np.arange(ny),
                        "x": np.arange(nx),
                    }
                )

                # initialize empty variable array
                shape = (0, len(self.equ), npe, nz, ny, nx)

                ds["data"] = (("time", "equations", "faces", "z", "y", "x"), np.zeros(shape, dtype=np.float64))

                ds["elev"] = (("faces", "z", "y", "x"), np.zeros((npe, nz, ny, nx), dtype=np.float64))

                ds["topo"] = (("faces", "y", "x"), np.zeros((npe, ny, nx), dtype=np.float64))

                # chunking
                ds = ds.chunk(
                    {
                        "time": 1,
                        "equations": 1,
                        "faces": 1,
                        "z": nz,
                        "y": ny,
                        "x": nx,
                    }
                )
        else:
            ds = xr.Dataset(
                coords={
                    "time": np.array([], dtype="datetime64[ns]"),
                    "equations": np.array(self.equ, dtype=object),
                    "faces": np.arange(npe),
                    "y": np.arange(ny),
                    "x": np.arange(nx),
                }
            )

            # initialize empty variable array
            shape = (0, len(self.equ), npe, ny, nx)

            ds["data"] = (
                ("time", "equations", "faces", "y", "x"),
                np.zeros(shape, dtype=np.float64),
            )

            # chunking
            ds = ds.chunk(
                {
                    "time": 1,
                    "equations": 1,
                    "faces": 1,
                    "y": ny,
                    "x": nx,
                }
            )

        if self.rank == 0:
            ds["time"].encoding = {
                "units": "seconds since 1800-01-01 00:00:00",
            }
            ds.to_zarr(filename, mode="w")

    # --------------------------------------------------
    def __write_result__(self, Q, step_id):
        #time_val = step_id.values
        time_val = (
            self.start_time
            + np.timedelta64(int(step_id * self.dt), "s")
        )
        if self.geometry.nk > 1: 
            nz = self.geometry.nk
        else:
            if self.geometry.z_levels:
                if type(self.geometry.z_levels) == int:
                    nz = self.geometry.z_levels
                else:
                    nz = len(self.geometry.z_levels)
            else:
                nz = 1

        ny = self.geometry.block_lat.shape[-2]
        nx = self.geometry.block_lat.shape[-1]

        t_index = self._append_time_step(
            time_val,
            nz,
            ny,
            nx,
        )

        if self.config.equations == "shallow_water":
            if nz > 1:
                for k in range(nz):
                    h = Q[idx_h, k, ...]

                    if self.topo is not None:
                        h = h + self.topo.hsurf

                    self.store_field_zarr_Zdim(self.geometry.to_single_block(h), 0, t_index, k)

                    u1 = Q[idx_hu1, k, ...] / h
                    u2 = Q[idx_hu2, k, ...] / h

                    u, v = self.geometry.contra2wind(u1, u2)

                    rv = relative_vorticity(u1, u2, self.metric, self.operators)

                    pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                    self.store_field_zarr_Zdim(self.geometry.to_single_block(u), 1, t_index, k)

                    self.store_field_zarr_Zdim(self.geometry.to_single_block(v), 2, t_index, k)

                    self.store_field_zarr_Zdim(self.geometry.to_single_block(rv), 3, t_index, k)

                    self.store_field_zarr_Zdim(self.geometry.to_single_block(pv), 4, t_index, k)

            else:

                h = Q[idx_h, :, :]

                if self.topo is not None:
                    h = h + self.topo.hsurf

                self.store_field_zarr(self.geometry.to_single_block(h), 0, t_index)

                u1 = Q[idx_hu1, :, :] / h
                u2 = Q[idx_hu2, :, :] / h

                u, v = self.geometry.contra2wind(u1, u2)

                rv = relative_vorticity(u1, u2, self.metric, self.operators)

                pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                self.store_field_zarr(self.geometry.to_single_block(u), 1, t_index)

                self.store_field_zarr(self.geometry.to_single_block(v), 2, t_index)

                self.store_field_zarr( self.geometry.to_single_block(rv), 3, t_index)

                self.store_field_zarr(self.geometry.to_single_block(pv), 4, t_index)

            return

        elif self.config.equations == "euler":

            rho = Q[idx_rho, ...]
            u1 = Q[idx_rho_u1, ...] / rho
            u2 = Q[idx_rho_u2, ...] / rho
            u3 = Q[idx_rho_w, ...] / rho
            theta = Q[idx_rho_theta, ...] / rho

            u, v, w = self.geometry.contra2wind_3d(u1, u2, u3, self.metric)

            p = p0 * (Q[idx_rho_theta] * Rd / p0) ** (cpd / cvd)
            
            for k in range(nz):

                self.store_field_zarr_Zdim(self.geometry.to_single_block(u), 0, t_index, k)

                self.store_field_zarr_Zdim(self.geometry.to_single_block(v), 1, t_index, k)

                self.store_field_zarr_Zdim(self.geometry.to_single_block(w), 2, t_index, k)

                self.store_field_zarr_Zdim(self.geometry.to_single_block(rho), 3, t_index, k)

                self.store_field_zarr_Zdim(self.geometry.to_single_block(theta), 4, t_index, k)

                self.store_field_zarr_Zdim(self.geometry.to_single_block(p), 5, t_index, k)

            return

    # --------------------------------------------------
    def __finalize__(self):
        return


    def _append_time_step(self, time_val, nz, ny, nx):

        if self.rank == 0:
            if nz != 1:
                if self.config.equations == "shallow_water":
                    dummy = xr.Dataset(
                        {
                            "data": (
                                ("time", "equations", "z", "faces", "y", "x"),
                                np.full(
                                    (1, len(self.equ), nz, 6, ny, nx),
                                    np.nan,
                                    dtype=np.float64,
                                ),
                            )
                        },
                        coords={
                            "time": np.array([time_val], dtype= "datetime64[s]"),
                            "equations": np.array(self.equ, dtype=object),
                            "z": np.arange(nz),
                            "faces": np.arange(6),
                            "y": np.arange(ny),
                            "x": np.arange(nx),
                        },
                    )
                if self.config.equations == "euler":
                    dummy = xr.Dataset(
                        {
                            "data": (
                                ("time", "equations", "faces", "z", "y", "x"),
                                np.full(
                                    (1, len(self.equ), 6, nz, ny, nx),
                                    np.nan,
                                    dtype=np.float64,
                                ),
                            )
                        },
                        coords={
                            "time": np.array([time_val], dtype= "datetime64[s]"),
                            "equations": np.array(self.equ, dtype=object),
                            "faces": np.arange(6),
                            "z": np.arange(nz),
                            "y": np.arange(ny),
                            "x": np.arange(nx),
                        },
                    )
            else:
                dummy = xr.Dataset(
                    {
                        "data": (
                            ("time", "equations", "faces", "y", "x"),
                            np.full(
                                (1, len(self.equ), 6, ny, nx),
                                np.nan,
                                dtype=np.float64,
                            ),
                        )
                    },
                    coords={
                        "time": np.array([time_val], dtype= "datetime64[s]"),
                        "equations": np.array(self.equ, dtype=object),
                        "faces": np.arange(6),
                        "y": np.arange(ny),
                        "x": np.arange(nx),
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

    def store_field_zarr(self, field, equation_idx, t_index):
        fields = self._gather_field(field, self.num_dim)

        if fields is None:
            return

        if self.rank == 0:
            root = zarr.open_group(
                self.filename,
                mode="r+",
            )

            for face_idx, face in enumerate(fields):
                root["data"][t_index, equation_idx, face_idx, :, :] = self.device.to_host(face)

    def store_field_zarr_Zdim(self, field, equation_idx, t_index, level_idx):
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
                    root["data"][t_index, equation_idx, level_idx, face_idx, :, :] = self.device.to_host(face)
                elif self.config.equations == "euler":
                    root["data"][t_index, equation_idx, face_idx, level_idx, :, :] = self.device.to_host(face[level_idx, :, :])

    def write_static_euler_fields(self):
        elev = self._gather_field(
            self.geometry.coordVec_latlon[2, :, :, :],
            3,
        )
        topo = self._gather_field(
            self.geometry.zbot,
            2,
        )

        if self.rank != 0:
            return

        root = zarr.open_group(
            self.filename,
            mode="r+",
        )

        root["elev"][:] = elev

        root["topo"][:] = topo
