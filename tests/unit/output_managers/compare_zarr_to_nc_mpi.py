from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import zarr
from mpi_test import MpiTestCase

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.simulation import Simulation
from wx_factory.wx_mpi import Conditional, SingleProcess


class CompareZarrToNcTestCase(MpiTestCase):
    def test_compare_zarr_to_nc(self):
        with SingleProcess(self.comm) as process, Conditional(process):
            directory = TemporaryDirectory(prefix="compare_zarr_nc_")
            process.return_value = directory.name
        output_dir = process.return_value
        try:
            for output_format in ("netcdf", "zarr"):
                config = Configuration(readfile("config/dcmip413.ini"), load_default_schema())
                config.num_elements_horizontal = config.num_elements_vertical = 1
                config.num_solpts = 2
                config.precision = "double"
                config.pytorch_device = "cpu"
                config.dt, config.t_end = 0.06, 0.12
                config.output_freq = 1
                config.output_dir = output_dir
                config.base_output_file = "comparison"
                config.output_format = output_format
                simulation = Simulation(config, context=self.context, quiet=True)
                for step in range(1, 3):
                    simulation.output.step(simulation.Q, step)
                simulation.output.finalize(0)
            failure = None
            if self.comm.rank == 0:
                try:
                    with netCDF4.Dataset(Path(output_dir) / "comparison.nc") as netcdf:
                        store = zarr.open_group(Path(output_dir) / "comparison.zarr", mode="r")
                        for name in ("rho", "U", "V", "W", "theta", "P", "q1", "q2", "lats", "lons", "elev", "topo"):
                            np.testing.assert_allclose(
                                store[name][:], netcdf[name][:], rtol=2e-14, atol=0, err_msg=name
                            )
                        np.testing.assert_allclose(netcdf["time"][:], [0, 0.06, 0.12], rtol=0, atol=1e-12)
                        elapsed = (store["time"][:] - store["time"][0]) / np.timedelta64(1, "s")
                        np.testing.assert_allclose(elapsed, netcdf["time"][:], rtol=0, atol=1e-12)
                        for name in ("q1", "q2"):
                            self.assertEqual(store[name].attrs["units"], netcdf[name].units)
                except Exception as error:  # noqa: BLE001 -- Propagate root-side failures to every rank.
                    failure = f"{type(error).__name__}: {error}"
            failure = self.comm.bcast(failure, root=0)
            self.assertIsNone(failure, failure)
        finally:
            self.comm.Barrier()
            if self.comm.rank == 0:
                directory.cleanup()
