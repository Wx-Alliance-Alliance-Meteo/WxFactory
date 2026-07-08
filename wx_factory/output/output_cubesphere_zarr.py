import numpy as np
import xarray as xr
from numcodecs import Blosc
from mpi4py import MPI
import os
import shutil


from .output_cubesphere import OutputCubesphere
from .diagnostic import potential_vorticity, relative_vorticity
from ..common.definitions import idx_h, idx_hu1, idx_hu2


class OutputCubesphereZarr(OutputCubesphere):

    equ = ["h", "U", "V", "RV", "PV"]

    def __init__(
        self,
        config,
        geometry,
        operators,
        device,
        metric,
        topo,
        dataset,
        process_topo,
        Q,
    ):
        super().__init__(config, geometry, operators, device, metric, topo, process_topo)
        self.dataset = dataset
        self.time_counter = {}
        self.panel_id = self.process_topology.my_panel
        self.max_time = len(self.dataset.time)

        # --- get real ERA5 time ---
        self.year = int(str(dataset.data["time"][0].values)[:4])
        self.filename = f"{self.output_dir}/{self.year}.zarr"
        self.marker_path = None

        # --- init once per year ---
        # check only on rank 0
        if self.rank == 0:
            exists = os.path.exists(self.filename)
            if exists:
                if self.is_zarr_in_progress(self.filename) == False:
                    print(f"Removing existing Zarr store: {self.filename}")
                    shutil.rmtree(self.filename)
            self._output_init(Q, self.filename)

        self.comm.Barrier()
        self.time_counter[self.year] = 0

    # --------------------------------------------------
    def _output_init(self, Q, filename):

        # gather coordinates
        lons = self.device.to_host(self.geometry.block_lon * 180 / np.pi)
        lats = self.device.to_host(self.geometry.block_lat * 180 / np.pi)

        nz = Q.shape[1] if Q.ndim == 5 else 1
        npe = 6
        ny = lons.shape[-2]
        nx = lats.shape[-1]

        # detect Z dimension
        nz = Q.shape[1] if Q.ndim == 5 else 1

        ds = xr.Dataset(
            coords={
                "time": np.arange(self.max_time),
                "equations": self.equ,
                "z": np.arange(nz),
                "faces": np.arange(npe),
                "y": np.arange(ny),
                "x": np.arange(nx),
            }
        )

        # initialize empty variable array
        shape = (self.max_time, len(self.equ), nz, npe, ny, nx)

        ds["data"] = (
            ("time", "equations", "z", "faces", "y", "x"),
            np.zeros(shape, dtype=np.float64),
        )

        # chunking
        ds = ds.chunk(
            {
                "time": 1,
                "equations": 1,
                "z": nz,
                "faces": 1,
                "y": ny // 2,
                "x": nx // 2,
            }
        )

        if self.rank == 0:
            ds.to_zarr(filename, mode="w")
            self.marker_path = self.create_inprogress_marker(filename)

    # --------------------------------------------------
    def __write_result__(self, Q, step_id):

        if Q.ndim == 5:
            nz = Q.shape[1]
        else:
            nz = 1

        fields_all_z = []
        for k in range(nz):

            if nz == 1:
                h = Q[idx_h, :, :]
            else:
                h = Q[idx_h, k, :, :]

            if self.topo is not None:
                h = h + self.topo.hsurf

            if nz == 1:
                u1 = Q[idx_hu1, :, :] / h
                u2 = Q[idx_hu2, :, :] / h
            else:
                u1 = Q[idx_hu1, k, :, :] / h
                u2 = Q[idx_hu2, k, :, :] / h

            u, v = self.geometry.contra2wind(u1, u2)

            rv = relative_vorticity(u1, u2, self.metric, self.operators)
            pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

            fields = [h, u, v, rv, pv]

            fields_all_z.append(fields)

        # shape: (z, variable, y, x)
        data = np.stack([np.stack(z_fields, axis=0) for z_fields in fields_all_z], axis=0)
        # data shape: (z, variable, elem_y, elem_x, solpts)

        z, nvar, ey, ex, npts = data.shape

        # reshape solpts → (sy, sx)
        # since num_solpts = 3 → 9 = 3×3
        ns = int(np.sqrt(npts))

        data = data.reshape(z, nvar, ey, ex, ns, ns)

        # move solpts into spatial grid
        data = data.transpose(0, 1, 2, 4, 3, 5)
        # now shape: (z, var, ey, sy, ex, sx)

        # merge element + solpts into full grid
        data = data.reshape(z, nvar, ey * ns, ex * ns)
        # now shape = (z, variable, y, x)

        # reorder to (variable, z, y, x)
        data = np.transpose(data, (1, 0, 2, 3))

        # add panel + time dims
        data = data[np.newaxis, :, :, np.newaxis, :, :]

        # final shape:
        # (time=1, variable=5, z, panel=1, y, x)

        # ---------------------------
        # WRITE REGION
        # ---------------------------

        t_index = self.time_counter[self.year]
        self.time_counter[self.year] += 1

        region = {
            "time": slice(t_index, t_index + 1),
            "equations": slice(0, len(self.equ)),
            "z": slice(0, nz),
            "faces": slice(self.panel_id, self.panel_id + 1),
            "y": slice(0, data.shape[-2]),
            "x": slice(0, data.shape[-1]),
        }

        ds_local = xr.Dataset({"data": (("time", "equations", "z", "faces", "y", "x"), data)})

        ds_local.to_zarr(self.filename, region=region)

    # --------------------------------------------------
    def __finalize__(self):

        if self.rank == 0:

            if hasattr(self, "marker_path") and os.path.exists(self.marker_path):
                os.remove(self.marker_path)

    def create_inprogress_marker(self, zarr_path):

        pid = os.getpid()

        marker_name = f".inprogress_" f"{pid}"

        marker_path = os.path.join(
            zarr_path,
            marker_name,
        )

        with open(marker_path, "w") as f:
            f.write("running\n")

        return marker_path

    def is_zarr_in_progress(self, path):

        for fname in os.listdir(path):

            if fname.startswith(".inprogress_"):
                return True

        return False

    def remove_stale_markers(self, zarr_path):

        for fname in os.listdir(zarr_path):

            if fname.startswith(".inprogress_"):

                os.remove(os.path.join(zarr_path, fname))
