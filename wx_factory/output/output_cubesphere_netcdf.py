import math
import time
from typing import List
import numpy as np

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from ..common.definitions import (
    idx_h,
    idx_hu1,
    idx_hu2,
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_w,
    idx_rho_theta,
    cpd,
    cvd,
    p0,
    Rd,
)
from ..common.configuration import Configuration
from ..device import Device
from ..geometry import CubedSphere, CubedSphere2D, CubedSphere3D, Metric2D, Metric3DTopo, DFROperators
from ..process_topology import ProcessTopology
from ..wx_mpi import SingleProcess, Conditional

from .diagnostic import potential_vorticity, relative_vorticity
from .output_cubesphere import OutputCubesphere


class OutputCubesphereNetcdf(OutputCubesphere):
    def __init__(
        self,
        config: Configuration,
        geometry: CubedSphere,
        operators: DFROperators,
        device: Device,
        metric: Metric2D | Metric3DTopo,
        topo,
        process_topo: ProcessTopology,
    ):
        super().__init__(config, geometry, operators, device, metric, topo, process_topo)

        self.ncfile = None
        self.filename = f"{self.output_dir}/{self.config.base_output_file}.nc"
        self.nz = None
        if config.case_number == -2:
            self.z_levels = config.z_levels

        """if config.output_freq > 0:
            self._output_init()"""
        self.initialized = False

    def _output_init(self, NZ):
        """Initialise the netCDF4 file."""

        # import here, so we don't need the module if not outputting
        import netCDF4

        # creating the netcdf file(s)
        with SingleProcess() as s, Conditional(s):
            self.ncfile = netCDF4.Dataset(self.filename, "w", format="NETCDF4")

        # create dimensions
        side = self.process_topology.num_lines_per_panel
        if self.config.equations == "shallow_water":
            nj, ni = self.geometry.block_shape
            ni *= side
            nj *= side
            grid_data = ("npe", "Xdim", "Ydim")
        elif self.config.equations == "euler":
            nk, nj, ni = self.geometry.nk, self.geometry.nj, self.geometry.ni
            nj *= side
            ni *= side
            grid_data = ("npe", "Zdim", "Xdim", "Ydim")
        else:
            raise ValueError(f"Unsupported equation type {self.config.equations}")

        grid_data2D = ("npe", "Xdim", "Ydim")

        if self.ncfile is not None:
            # write general attributes
            self.ncfile.history = "Created " + time.ctime(time.time())
            self.ncfile.description = "WxFactory Model"
            self.ncfile.details = "Cubed-sphere coordinates, Gauss-Legendre collocated grid"

            self.ncfile.createDimension("time", None)  # unlimited
            npe = 6
            self.ncfile.createDimension("npe", npe)
            self.ncfile.createDimension("Ydim", ni)
            self.ncfile.createDimension("Xdim", nj)

            if self.config.equations == "shallow_water" and self.config.case_number == -2:

                self.ncfile.createDimension("Zdim", NZ)

                zzz = self.ncfile.createVariable("Zdim", numpy.float64, ("Zdim",))
                zzz.long_name = "Zdim"
                zzz.axis = "Z"
                zzz.units = "m"

                if self.rank == 0:
                    if hasattr(self, "z_levels"):
                        zzz[:] = self.z_levels
                    else:
                        zzz[:] = numpy.arange(NZ)

            # create time axis
            tme = self.ncfile.createVariable("time", numpy.float64, ("time",))
            if self.config.case_number == -2:
                tme.units = "hours since 1800-01-01 00:00:00"
                tme.calendar = "standard"
            else:
                tme.units = "hours since 1800-01-01"
            tme.long_name = "time"

            # create tiles axis
            tile = self.ncfile.createVariable("npe", "i4", ("npe"))
            tile.grads_dim = "e"
            tile.standard_name = "tile"
            tile.long_name = "cubed-sphere tile"
            tile.axis = "e"

            # create latitude axis
            yyy = self.ncfile.createVariable("Ydim", numpy.float64, ("Ydim"))
            yyy.long_name = "Ydim"
            yyy.axis = "Y"
            yyy.units = "radians_north"

            # create longitude axis
            xxx = self.ncfile.createVariable("Xdim", numpy.float64, ("Xdim"))
            xxx.long_name = "Xdim"
            xxx.axis = "X"
            xxx.units = "radians_east"

            if self.config.equations == "euler":
                self.ncfile.createDimension("Zdim", nk)
                zzz = self.ncfile.createVariable("Zdim", numpy.float64, ("Zdim"))
                zzz.long_name = "Zdim"
                zzz.axis = "Z"
                zzz.units = "m"

            # create variable array
            lat = self.ncfile.createVariable("lats", numpy.float64, grid_data2D)
            lat.long_name = "latitude"
            lat.units = "degrees_north"

            lon = self.ncfile.createVariable("lons", numpy.dtype("double").char, grid_data2D)
            lon.long_name = "longitude"
            lon.units = "degrees_east"

            if self.config.equations == "shallow_water":

                if self.config.case_number == -2:
                    dims = ("time", "Zdim") + grid_data
                else:
                    dims = ("time",) + grid_data

                hhh = self.ncfile.createVariable("h", numpy.dtype("double").char, dims)
                hhh.long_name = "fluid height"
                hhh.units = "m"
                hhh.coordinates = "lons lats"
                hhh.grid_mapping = "cubed_sphere"

                if self.config.case_number >= 2 or self.config.case_number == -1 or self.config.case_number == -2:
                    uuu = self.ncfile.createVariable("U", numpy.dtype("double").char, dims)
                    uuu.long_name = "eastward_wind"
                    uuu.units = "m s-1"
                    uuu.standard_name = "eastward_wind"
                    uuu.coordinates = "lons lats"
                    uuu.grid_mapping = "cubed_sphere"

                    vvv = self.ncfile.createVariable("V", numpy.dtype("double").char, dims)
                    vvv.long_name = "northward_wind"
                    vvv.units = "m s-1"
                    vvv.standard_name = "northward_wind"
                    vvv.coordinates = "lons lats"
                    vvv.grid_mapping = "cubed_sphere"

                    drv = self.ncfile.createVariable("RV", numpy.dtype("double").char, dims)
                    drv.long_name = "Relative vorticity"
                    drv.units = "1/(m s)"
                    drv.standard_name = "Relative vorticity"
                    drv.coordinates = "lons lats"
                    drv.grid_mapping = "cubed_sphere"

                    dpv = self.ncfile.createVariable("PV", numpy.dtype("double").char, dims)
                    dpv.long_name = "Potential vorticity"
                    dpv.units = "1/(m s)"
                    dpv.standard_name = "Potential vorticity"
                    dpv.coordinates = "lons lats"
                    dpv.grid_mapping = "cubed_sphere"

            elif self.config.equations == "euler":
                elev = self.ncfile.createVariable("elev", numpy.dtype("double").char, grid_data)
                elev.long_name = "Elevation"
                elev.units = "m"
                elev.standard_name = "Elevation"
                elev.coordinates = "lons lats"
                elev.grid_mapping = "cubed_sphere"

                topo = self.ncfile.createVariable("topo", numpy.dtype("double").char, grid_data2D)
                topo.long_name = "Topopgraphy"
                topo.units = "m"
                topo.standard_name = "Topography"
                topo.coordinates = "lons lats"
                topo.grid_mapping = "cubed_sphere"

                uuu = self.ncfile.createVariable("U", numpy.dtype("double").char, ("time",) + grid_data)
                uuu.long_name = "eastward_wind"
                uuu.units = "m s-1"
                uuu.standard_name = "eastward_wind"
                uuu.coordinates = "lons lats"
                uuu.grid_mapping = "cubed_sphere"

                vvv = self.ncfile.createVariable("V", numpy.dtype("double").char, ("time",) + grid_data)
                vvv.long_name = "northward_wind"
                vvv.units = "m s-1"
                vvv.standard_name = "northward_wind"
                vvv.coordinates = "lons lats"
                vvv.grid_mapping = "cubed_sphere"

                www = self.ncfile.createVariable("W", numpy.dtype("double").char, ("time",) + grid_data)
                www.long_name = "upward_air_velocity"
                www.units = "m s-1"
                www.standard_name = "upward_air_velocity"
                www.coordinates = "lons lats"
                www.grid_mapping = "cubed_sphere"

                density = self.ncfile.createVariable("rho", numpy.dtype("double").char, ("time",) + grid_data)
                density.long_name = "air_density"
                density.units = "kg m-3"
                density.standard_name = "air_density"
                density.coordinates = "lons lats"
                density.grid_mapping = "cubed_sphere"

                potential_temp = self.ncfile.createVariable("theta", numpy.dtype("double").char, ("time",) + grid_data)
                potential_temp.long_name = "air_potential_temperature"
                potential_temp.units = "K"
                potential_temp.standard_name = "air_potential_temperature"
                potential_temp.coordinates = "lons lats"
                potential_temp.grid_mapping = "cubed_sphere"

                press = self.ncfile.createVariable("P", numpy.dtype("double").char, ("time",) + grid_data)
                press.long_name = "air_pressure"
                press.units = "Pa"
                press.standard_name = "air_pressure"
                press.coordinates = "lons lats"
                press.grid_mapping = "cubed_sphere"

                if self.config.case_number == 11 or self.config.case_number == 12:
                    q1 = self.ncfile.createVariable("q1", numpy.dtype("double").char, ("time",) + grid_data)
                    q1.long_name = "q1"
                    q1.units = "kg m-3"
                    q1.standard_name = "Tracer q1"
                    q1.coordinates = "lons lats"
                    q1.grid_mapping = "cubed_sphere"

                if self.config.case_number == 11:
                    q2 = self.ncfile.createVariable("q2", numpy.dtype("double").char, ("time",) + grid_data)
                    q2.long_name = "q2"
                    q2.units = "kg m-3"
                    q2.standard_name = "Tracer q2"
                    q2.coordinates = "lons lats"
                    q2.grid_mapping = "cubed_sphere"

                    q3 = self.ncfile.createVariable("q3", numpy.dtype("double").char, ("time",) + grid_data)
                    q3.long_name = "q3"
                    q3.units = "kg m-3"
                    q3.standard_name = "Tracer q3"
                    q3.coordinates = "lons lats"
                    q3.grid_mapping = "cubed_sphere"

                    q4 = self.ncfile.createVariable("q4", numpy.dtype("double").char, ("time",) + grid_data)
                    q4.long_name = "q4"
                    q4.units = "kg m-3"
                    q4.standard_name = "Tracer q4"
                    q4.coordinates = "lons lats"
                    q4.grid_mapping = "cubed_sphere"

        to_host = lambda a: self.device.to_host(a) if a is not None else None

        panel_x = to_host(self._gather_panel(self.geometry.x1[...]))
        panel_y = to_host(self._gather_panel(self.geometry.x2[...]))

        if self.rank == 0:
            xxx[:] = panel_x
            yyy[:] = panel_y
            if self.config.equations == "euler":
                # No gathering needed for vertical coords
                # FIXME: With mapped coordinates, x3/height is a truly 3D coordinate
                zzz[:] = to_host(self.geometry.x3[:, 0, 0])

        self.comm.barrier()
        lons = to_host(self._gather_field(self.geometry.block_lon * 180 / math.pi, 2))
        lats = to_host(self._gather_field(self.geometry.block_lat * 180 / math.pi, 2))
        if self.config.equations == "euler":
            elevs = to_host(self._gather_field(self.geometry.coordVec_latlon[2, :, :, :], 3))
            topos = to_host(self._gather_field(self.geometry.zbot[:, :], 2))

        if self.rank == 0:
            for i in range(6):
                tile[i] = i
                lon[i, :, :] = lons[i]
                lat[i, :, :] = lats[i]
            if self.config.equations == "euler":
                for i in range(6):
                    elev[i, :, :, :] = elevs[i]
                    topo[i, :, :] = topos[i]

    def store_field_Zdim(self, field, name: str, time_idx: int, level_idx: int):
        fields = self._gather_field(field, self.num_dim)

        if fields is None:
            return

        to_host = self.device.to_host

        for i, f in enumerate(fields):
            self.ncfile[name][time_idx, level_idx, i, :, :] = to_host(f)

    def __write_result__(self, Q, step_id):

        if not self.initialized:

            if Q.ndim == 5:
                self.nz = Q.shape[0]
            else:
                self.nz = 1

            self._output_init(self.nz)

            self.initialized = True

        geom = self.geometry

        if self.rank == 0:
            idx = len(self.ncfile["time"])
        else:
            idx = 0

        if isinstance(geom, CubedSphere2D):  # Shallow water

            if Q.ndim == 5:
                for k in range(self.nz):

                    h = Q[k, idx_h, ...]

                    if self.topo is not None:
                        h = h + self.topo.hsurf

                    field_block = geom.to_single_block(h)

                    self.store_field_Zdim(field_block, "h", idx, k)

                    if self.config.case_number >= 2 or self.config.case_number in [-1, -2]:

                        u1 = Q[k, idx_hu1, ...] / h
                        u2 = Q[k, idx_hu2, ...] / h

                        u, v = geom.contra2wind(u1, u2)

                        self.store_field_Zdim(geom.to_single_block(u), "U", idx, k)
                        self.store_field_Zdim(geom.to_single_block(v), "V", idx, k)

                        rv = relative_vorticity(u1, u2, self.metric, self.operators)
                        pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                        self.store_field_Zdim(geom.to_single_block(rv), "RV", idx, k)
                        self.store_field_Zdim(geom.to_single_block(pv), "PV", idx, k)

            else:
                h = Q[idx_h, :, :]
                if self.topo is not None:
                    h = Q[idx_h, :, :] + self.topo.hsurf

                self.store_field(geom.to_single_block(h), "h", idx)

                if self.config.case_number >= 2 or self.config.case_number in [-1, -2]:

                    u1 = Q[idx_hu1, :, :] / h
                    u2 = Q[idx_hu2, :, :] / h

                    u, v = geom.contra2wind(u1, u2)

                    rv = relative_vorticity(u1, u2, self.metric, self.operators)
                    pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                    self.store_field(geom.to_single_block(u), "U", idx)
                    self.store_field(geom.to_single_block(v), "V", idx)
                    self.store_field(geom.to_single_block(rv), "RV", idx)
                    self.store_field(geom.to_single_block(pv), "PV", idx)

        elif isinstance(geom, CubedSphere3D):  # Euler equations
            rho = Q[idx_rho, ...]
            u1 = Q[idx_rho_u1, ...] / rho
            u2 = Q[idx_rho_u2, ...] / rho
            u3 = Q[idx_rho_w, ...] / rho
            theta = Q[idx_rho_theta, ...] / rho

            u, v, w = geom.contra2wind_3d(u1, u2, u3, self.metric)

            self.store_field(geom.to_single_block(rho), "rho", idx)
            self.store_field(geom.to_single_block(u), "U", idx)
            self.store_field(geom.to_single_block(v), "V", idx)
            self.store_field(geom.to_single_block(w), "W", idx)
            self.store_field(geom.to_single_block(theta), "theta", idx)
            self.store_field(geom.to_single_block(p0 * (Q[idx_rho_theta] * Rd / p0) ** (cpd / cvd)), "P", idx)

            if self.config.case_number == 11 or self.config.case_number == 12:
                self.store_field(geom.to_single_block(Q[5, ...] / rho), "q1", idx)

            if self.config.case_number == 11:
                for i in [6, 7, 8]:
                    self.store_field(geom.to_single_block(Q[i, ...] / rho), f"q{i-4}", idx)

        else:
            raise ValueError(f"Unknown class for geom: {geom}")

        if self.rank == 0:
            if self.config.case_number == -2:

                time_val = step_id

                epoch = np.datetime64("1800-01-01T00:00:00")
                hours = (time_val - epoch) / np.timedelta64(1, "h")

                self.ncfile["time"][idx] = hours
            else:
                self.ncfile["time"][idx] = step_id * self.config.dt

    def __finalize__(self):
        """Finalise the output netCDF4 file."""
        if self.rank == 0 and self.ncfile is not None:
            self.ncfile.close()

    def store_field(self, field: NDArray, name: str, step_id: int) -> None:
        """Store data in the open netcdf file."""
        fields = self._gather_field(field, self.num_dim)
        if fields is not None:
            to_host = self.device.to_host
            for i, f in enumerate(fields):
                self.ncfile[name][step_id, i] = to_host(f)
