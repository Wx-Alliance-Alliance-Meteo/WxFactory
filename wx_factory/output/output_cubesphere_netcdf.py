import math
import time
from typing import List

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from common.definitions import (
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
from common.configuration import Configuration
from device import Device
from geometry import CubedSphere, CubedSphere2D, CubedSphere3D, Metric2D, Metric3DTopo, DFROperators
from wx_mpi import ProcessTopology

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
        self.netcdf_serial = False
        self.filename = f"{self.output_dir}/{self.config.base_output_file}.nc"

        if config.output_freq > 0:
            self._output_init()

    def _output_init(self):
        """Initialise the netCDF4 file."""

        # import here, so we don't need the module if not outputting
        import netCDF4

        # creating the netcdf file(s)
        try:
            self.ncfile = netCDF4.Dataset(self.filename, "w", format="NETCDF4", parallel=True)
        except ValueError:
            self.netcdf_serial = True
            if self.rank == 0:
                print(f"WARNING: Unable to open a netCDF4 file in parallel mode. Doing it serially instead", flush=True)
                try:
                    self.ncfile = netCDF4.Dataset(self.filename, "w", format="NETCDF4")
                except:
                    print(f"unable to create file serially...", flush=True)
                    raise

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
            self.ncfile.description = "GEF Model"
            self.ncfile.details = "Cubed-sphere coordinates, Gauss-Legendre collocated grid"

            self.ncfile.createDimension("time", None)  # unlimited
            npe = 6
            self.ncfile.createDimension("npe", npe)
            self.ncfile.createDimension("Ydim", ni)
            self.ncfile.createDimension("Xdim", nj)

            # create time axis
            tme = self.ncfile.createVariable("time", numpy.float64, ("time",))
            tme.units = "hours since 1800-01-01"
            tme.long_name = "time"
            tme.set_collective(not self.netcdf_serial)

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

                hhh = self.ncfile.createVariable("h", numpy.dtype("double").char, ("time",) + grid_data)
                hhh.long_name = "fluid height"
                hhh.units = "m"
                hhh.coordinates = "lons lats"
                hhh.grid_mapping = "cubed_sphere"
                hhh.set_collective(not self.netcdf_serial)

                if self.config.case_number >= 2:
                    uuu = self.ncfile.createVariable("U", numpy.dtype("double").char, ("time",) + grid_data)
                    uuu.long_name = "eastward_wind"
                    uuu.units = "m s-1"
                    uuu.standard_name = "eastward_wind"
                    uuu.coordinates = "lons lats"
                    uuu.grid_mapping = "cubed_sphere"
                    uuu.set_collective(not self.netcdf_serial)

                    vvv = self.ncfile.createVariable("V", numpy.dtype("double").char, ("time",) + grid_data)
                    vvv.long_name = "northward_wind"
                    vvv.units = "m s-1"
                    vvv.standard_name = "northward_wind"
                    vvv.coordinates = "lons lats"
                    vvv.grid_mapping = "cubed_sphere"
                    vvv.set_collective(not self.netcdf_serial)

                    drv = self.ncfile.createVariable("RV", numpy.dtype("double").char, ("time",) + grid_data)
                    drv.long_name = "Relative vorticity"
                    drv.units = "1/(m s)"
                    drv.standard_name = "Relative vorticity"
                    drv.coordinates = "lons lats"
                    drv.grid_mapping = "cubed_sphere"
                    drv.set_collective(not self.netcdf_serial)

                    dpv = self.ncfile.createVariable("PV", numpy.dtype("double").char, ("time",) + grid_data)
                    dpv.long_name = "Potential vorticity"
                    dpv.units = "1/(m s)"
                    dpv.standard_name = "Potential vorticity"
                    dpv.coordinates = "lons lats"
                    dpv.grid_mapping = "cubed_sphere"
                    dpv.set_collective(not self.netcdf_serial)

            elif self.config.equations == "euler":
                elev = self.ncfile.createVariable("elev", numpy.dtype("double").char, grid_data)
                elev.long_name = "Elevation"
                elev.units = "m"
                elev.standard_name = "Elevation"
                elev.coordinates = "lons lats"
                elev.grid_mapping = "cubed_sphere"
                elev.set_collective(not self.netcdf_serial)

                topo = self.ncfile.createVariable("topo", numpy.dtype("double").char, grid_data2D)
                topo.long_name = "Topopgraphy"
                topo.units = "m"
                topo.standard_name = "Topography"
                topo.coordinates = "lons lats"
                topo.grid_mapping = "cubed_sphere"
                topo.set_collective(not self.netcdf_serial)

                uuu = self.ncfile.createVariable("U", numpy.dtype("double").char, ("time",) + grid_data)
                uuu.long_name = "eastward_wind"
                uuu.units = "m s-1"
                uuu.standard_name = "eastward_wind"
                uuu.coordinates = "lons lats"
                uuu.grid_mapping = "cubed_sphere"
                uuu.set_collective(not self.netcdf_serial)

                vvv = self.ncfile.createVariable("V", numpy.dtype("double").char, ("time",) + grid_data)
                vvv.long_name = "northward_wind"
                vvv.units = "m s-1"
                vvv.standard_name = "northward_wind"
                vvv.coordinates = "lons lats"
                vvv.grid_mapping = "cubed_sphere"
                vvv.set_collective(not self.netcdf_serial)

                www = self.ncfile.createVariable("W", numpy.dtype("double").char, ("time",) + grid_data)
                www.long_name = "upward_air_velocity"
                www.units = "m s-1"
                www.standard_name = "upward_air_velocity"
                www.coordinates = "lons lats"
                www.grid_mapping = "cubed_sphere"
                www.set_collective(not self.netcdf_serial)

                density = self.ncfile.createVariable("rho", numpy.dtype("double").char, ("time",) + grid_data)
                density.long_name = "air_density"
                density.units = "kg m-3"
                density.standard_name = "air_density"
                density.coordinates = "lons lats"
                density.grid_mapping = "cubed_sphere"
                density.set_collective(not self.netcdf_serial)

                potential_temp = self.ncfile.createVariable("theta", numpy.dtype("double").char, ("time",) + grid_data)
                potential_temp.long_name = "air_potential_temperature"
                potential_temp.units = "K"
                potential_temp.standard_name = "air_potential_temperature"
                potential_temp.coordinates = "lons lats"
                potential_temp.grid_mapping = "cubed_sphere"
                potential_temp.set_collective(not self.netcdf_serial)

                press = self.ncfile.createVariable("P", numpy.dtype("double").char, ("time",) + grid_data)
                press.long_name = "air_pressure"
                press.units = "Pa"
                press.standard_name = "air_pressure"
                press.coordinates = "lons lats"
                press.grid_mapping = "cubed_sphere"
                press.set_collective(not self.netcdf_serial)

                if self.config.case_number == 11 or self.config.case_number == 12:
                    q1 = self.ncfile.createVariable("q1", numpy.dtype("double").char, ("time",) + grid_data)
                    q1.long_name = "q1"
                    q1.units = "kg m-3"
                    q1.standard_name = "Tracer q1"
                    q1.coordinates = "lons lats"
                    q1.grid_mapping = "cubed_sphere"
                    q1.set_collective(not self.netcdf_serial)

                if self.config.case_number == 11:
                    q2 = self.ncfile.createVariable("q2", numpy.dtype("double").char, ("time",) + grid_data)
                    q2.long_name = "q2"
                    q2.units = "kg m-3"
                    q2.standard_name = "Tracer q2"
                    q2.coordinates = "lons lats"
                    q2.grid_mapping = "cubed_sphere"
                    q2.set_collective(not self.netcdf_serial)

                    q3 = self.ncfile.createVariable("q3", numpy.dtype("double").char, ("time",) + grid_data)
                    q3.long_name = "q3"
                    q3.units = "kg m-3"
                    q3.standard_name = "Tracer q3"
                    q3.coordinates = "lons lats"
                    q3.grid_mapping = "cubed_sphere"
                    q3.set_collective(not self.netcdf_serial)

                    q4 = self.ncfile.createVariable("q4", numpy.dtype("double").char, ("time",) + grid_data)
                    q4.long_name = "q4"
                    q4.units = "kg m-3"
                    q4.standard_name = "Tracer q4"
                    q4.coordinates = "lons lats"
                    q4.grid_mapping = "cubed_sphere"
                    q4.set_collective(not self.netcdf_serial)

        prepare = self.device.to_host

        panel_x = self._gather_panel(prepare(self.geometry.x1[...]))
        panel_y = self._gather_panel(prepare(self.geometry.x2[...]))

        if self.rank == 0:
            xxx[:] = panel_x
            yyy[:] = panel_y
            if self.config.equations == "euler":
                # No gathering needed for vertical coords
                # FIXME: With mapped coordinates, x3/height is a truly 3D coordinate
                zzz[:] = prepare(self.geometry.x3[:, 0, 0])

        if self.netcdf_serial:
            lons = self._gather_field(prepare(self.geometry.block_lon * 180 / math.pi))
            lats = self._gather_field(prepare(self.geometry.block_lat * 180 / math.pi))
            if self.config.equations == "euler":
                elevs = self._gather_field(prepare(self.geometry.coordVec_latlon[2, :, :, :]))
                topos = self._gather_field(prepare(self.geometry.zbot[:, :]))

            if self.rank == 0:
                for i in range(6):
                    tile[i] = i
                    lon[i, :, :] = lons[i]
                    lat[i, :, :] = lats[i]
                if self.config.equations == "euler":
                    for i in range(6):
                        elev[i, :, :, :] = elevs[i]
                        topo[i, :, :] = topos[i]

        else:
            panel_lon = self._gather_panel(prepare(self.geometry.lon * 180 / math.pi))
            panel_lat = self._gather_panel(prepare(self.geometry.lat * 180 / math.pi))
            if self.config.equations == "euler":
                panel_elev = self._gather_panel(prepare(self.geometry.coordVec_latlon[2, :, :, :]))
                panel_topo = self._gather_panel(prepare(self.geometry.zbot[:, :]))
            if panel_lon is not None:
                root_rank = self.process_topology.panel_roots_comm.rank
                tile[root_rank] = root_rank
                lon[root_rank, :, :] = panel_lon
                lat[root_rank, :, :] = panel_lat
                if self.config.equations == "euler":
                    elev[root_rank, :, :, :] = panel_elev
                    topo[root_rank, :, :] = panel_topo

    def __write_result__(self, Q, step_id):
        prepare = self.device.to_host
        geom = self.geometry

        idx = 0
        if self.ncfile is not None:
            idx = len(self.ncfile["time"])
            self.ncfile["time"][idx] = step_id * self.config.dt

        if isinstance(geom, CubedSphere2D):  # Shallow water

            # Unpack physical variables
            h = Q[idx_h, :, :]
            if self.topo is not None:
                h += self.topo.hsurf
            self.store_field(geom.to_single_block(prepare(h)), "h", idx)

            if self.config.case_number >= 2:
                u1 = Q[idx_hu1, :, :] / h
                u2 = Q[idx_hu2, :, :] / h
                u, v = geom.contra2wind(u1, u2)
                rv = relative_vorticity(u1, u2, self.metric, self.operators)
                pv = potential_vorticity(h, u1, u2, self.metric, self.operators)

                self.store_field(prepare(geom.to_single_block(u)), "U", idx)
                self.store_field(prepare(geom.to_single_block(v)), "V", idx)
                self.store_field(prepare(geom.to_single_block(rv)), "RV", idx)
                self.store_field(prepare(geom.to_single_block(pv)), "PV", idx)

        elif isinstance(geom, CubedSphere3D):  # Euler equations
            rho = Q[idx_rho, ...]
            u1 = Q[idx_rho_u1, ...] / rho
            u2 = Q[idx_rho_u2, ...] / rho
            u3 = Q[idx_rho_w, ...] / rho
            theta = Q[idx_rho_theta, ...] / rho

            u, v, w = geom.contra2wind_3d(u1, u2, u3, self.metric)

            self.store_field(geom.to_single_block(prepare(rho)), "rho", idx)
            self.store_field(geom.to_single_block(prepare(u)), "U", idx)
            self.store_field(geom.to_single_block(prepare(v)), "V", idx)
            self.store_field(geom.to_single_block(prepare(w)), "W", idx)
            self.store_field(geom.to_single_block(prepare(theta)), "theta", idx)
            self.store_field(geom.to_single_block(prepare(p0 * (Q[idx_rho_theta] * Rd / p0) ** (cpd / cvd))), "P", idx)

            if self.config.case_number == 11 or self.config.case_number == 12:
                self.store_field(geom.to_single_block(prepare(Q[5, ...] / rho)), "q1", idx)

            if self.config.case_number == 11:
                for i in [6, 7, 8]:
                    self.store_field(geom.to_single_block(prepare(Q[i, ...] / rho)), f"q{i-4}", idx)

        else:
            raise ValueError(f"Unknown class for geom: {geom}")

    def __finalize__(self):
        """Finalise the output netCDF4 file."""
        if self.rank == 0 and self.ncfile is not None:
            self.ncfile.close()

    def store_field(self, field: NDArray, name: str, step_id: int) -> None:
        """Store data in a given file.

        If the netcdf_serial option is activated, this will gather the data on a single PE, and
        only that PE will perform the write operation.
        """
        if self.netcdf_serial:
            fields = self._gather_field(field)
            if fields is not None:
                for i, f in enumerate(fields):
                    self.ncfile[name][step_id, i] = f
        else:
            panel_field = self._gather_panel(field)
            if panel_field is not None:
                root_rank = self.process_topology.panel_roots_comm.rank
                self.ncfile[name][step_id, root_rank] = panel_field
