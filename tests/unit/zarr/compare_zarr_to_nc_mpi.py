import os
import shutil
import tempfile

import numpy as np
from numpy.typing import NDArray
import xarray as xr
from mpi4py import MPI

from mpi_test import MpiTestCase
from wx_factory.output import InputManager
from wx_factory.simulation import Simulation
from wx_factory.wx_mpi import SingleProcess, Conditional
from wx_factory.geometry import (
    Cartesian2D,
    CubedSphere,
    CubedSphere3D,
    CubedSphere2D,
    Geometry,
    DFROperators,
    Metric2D,
    Metric3DTopo,
)
from wx_factory.step_hooks import ScharMountainHook
from wx_factory.output.output_manager import OutputManager
from wx_factory.output.output_cartesian import OutputCartesian
from wx_factory.output.output_cubesphere_netcdf import OutputCubesphereNetcdf
from wx_factory.output.output_cubesphere_fst import OutputCubesphereFst
from wx_factory.output.output_cubesphere_zarr import OutputCubesphereZarr
from wx_factory.process_topology import ProcessTopology
from wx_factory.device import Device, CpuDevice, CudaDevice, PytorchDevice


class CompareZarrToNcTestCase(MpiTestCase):

    def __init__(self, num_procs, methodName="runTest", optional=False):
        super().__init__(num_procs, methodName, optional)
        self.state_dir = "tests/data/unit/zarr"
        self.metric = None
        self.topography = None
        self.comm = MPI.COMM_WORLD
        self.config = None
        self.device = None
        self.operators_real = None
        self.process_topo = None
        self.num_elements_horizontal = None
        self.total_num_elements_horizontal = None
        self.num_solpts = None
        self.step_hooks = {}
        self.Q = None

    def _find_pairs(self):
        files = [f for f in os.listdir(self.state_dir) if f.startswith("state_vector_") and f.endswith(".npy")]

        nc_files = {}
        zarr_files = {}

        for f in files:
            if f.endswith("_nc.npy"):
                key = f[:-7]  # remove "_nc.npy"
                nc_files[key] = os.path.join(self.state_dir, f)

            elif f.endswith("_zarr.npy"):
                key = f[:-9]  # remove "_zarr.npy"
                zarr_files[key] = os.path.join(self.state_dir, f)

        common_keys = sorted(set(nc_files.keys()) & set(zarr_files.keys()))

        return [(nc_files[k], zarr_files[k]) for k in common_keys]

    def _generate_output(self, state_file):
        self.config, global_state = InputManager.read_config_from_save_file(
            state_file,
            self.comm,
        )

        with SingleProcess(self.comm) as s, Conditional(s):
            s.return_value = tempfile.mkdtemp(prefix="compare_zarr_nc_")

        tmp_output_dir = s.return_value

        self.config.output_dir = tmp_output_dir

        if self.config.grid_file != "":
            self.num_elements_horizontal, self.num_solpts, self.lambda0, self.phi0, self.alpha0 = (
                InputManager.read_grid_params(self.config.grid_file, self.comm)
            )
        else:
            self.num_elements_horizontal = self.config.num_elements_horizontal
            self.num_solpts = self.config.num_solpts
            if self.config.grid_type == "cubed_sphere":
                self.lambda0 = self.config.lambda0
                self.phi0 = self.config.phi0
                self.alpha0 = self.config.alpha0

        self.allowed_pe_counts = (
            [
                i**2 * 6
                for i in range(1, max(self.num_elements_horizontal // 2 + 1, 2))
                if (self.num_elements_horizontal % i) == 0
            ]
            if self.config.grid_file != "" or self.config.grid_type == "cubed_sphere"
            else [1]
        )
        self._adjust_num_elements()
        self.device = self._make_device()
        self.geometry = self._create_geometry()
        self.operators_real = DFROperators(self.geometry, self.config, self.device)

        if self.config.equations == "euler" and isinstance(self.geometry, CubedSphere3D):
            self.metric = Metric3DTopo(self.geometry, self.operators_real)

        elif self.config.equations == "euler" and isinstance(self.geometry, Cartesian2D):
            self.metric = None

        elif self.config.equations == "shallow_water" and isinstance(self.geometry, CubedSphere2D):
            self.metric = Metric2D(self.geometry)
        if self.comm.rank == 0:
            num_dim = len(global_state.shape) - 2
        else:
            num_dim = None
        num_dim = self.comm.bcast(num_dim, root=0)
        self.Q = self.process_topo.distribute_cube(
            global_state,
            num_dim,
        )
        if len(self.Q.shape) == 5:
            self.geometry.z_levels = self.Q.shape[1]
        else:
            self.geometry.z_levels = 1
        self.output = self._create_output_manager()
        self.output.step(self.Q, 0)
        self.output.finalize(0.0)

        return self.config, tmp_output_dir

    def _compare_outputs(self, nc_file, zarr_store):

        ds_nc = xr.open_dataset(nc_file)
        ds_zarr = xr.open_zarr(zarr_store)

        try:
            self.assertIn(
                "data",
                ds_zarr.data_vars,
                "Variable 'data' not found in Zarr dataset",
            )
            equations = [str(v) for v in ds_zarr["equations"].values.tolist()]

            for equation_index, variable_name in enumerate(equations):
                self.assertIn(
                    variable_name,
                    ds_nc.data_vars,
                    f"Variable '{variable_name}' not found in NetCDF",
                )
                nc_values = ds_nc[variable_name].values

                zarr_values = ds_zarr["data"].isel(equations=equation_index).values

                self.assertEqual(
                    nc_values.shape,
                    zarr_values.shape,
                    (
                        f"Shape mismatch for '{variable_name}'. "
                        f"NetCDF={nc_values.shape}, "
                        f"Zarr={zarr_values.shape}"
                    ),
                )
                if not np.array_equal(
                    nc_values,
                    zarr_values,
                ):

                    diff = np.abs(nc_values - zarr_values)

                    mismatch_locations = np.argwhere(nc_values != zarr_values)

                    max_diff = np.max(diff)

                    report = [
                        f"Variable '{variable_name}' differs.",
                        f"Number of mismatches : {len(mismatch_locations)}",
                        f"Maximum difference   : {max_diff:.16e}",
                        "",
                        "First mismatches:",
                    ]
                    self.fail("\n".join(report))

        finally:
            ds_nc.close()
            ds_zarr.close()

    def test_compare_zarr_to_nc(self):
        try:
            pairs = self._find_pairs()
            if len(pairs) == 0:
                self.skipTest(f"No state-vector pairs found in {self.state_dir}")

            for nc_state_file, zarr_state_file in pairs:

                config_nc, tmp_nc_dir = self._generate_output(nc_state_file)

                config_zarr, tmp_zarr_dir = self._generate_output(zarr_state_file)

                try:

                    nc_file = f"{tmp_nc_dir}/" f"{config_nc.base_output_file}.nc"

                    zarr_store = f"{tmp_zarr_dir}/" f"{config_zarr.base_output_file}.zarr"

                    if not os.path.exists(nc_file):
                        self.fail(f"Expected NetCDF output not found: " f"{nc_file}")
                        print("Pas de NetCDF")

                    if not os.path.exists(zarr_store):
                        self.fail(f"Expected Zarr output not found: " f"{zarr_store}")
                        print("Pas de Zarr")

                    self._compare_outputs(
                        nc_file,
                        zarr_store,
                    )

                finally:
                    self.comm.Barrier()
                    if self.comm.rank == 0:
                        shutil.rmtree(
                            tmp_nc_dir,
                            ignore_errors=True,
                        )

                        shutil.rmtree(
                            tmp_zarr_dir,
                            ignore_errors=True,
                        )
                    self.comm.Barrier()
        except Exception as e:
            print(
                f"Rank {self.comm.rank} failed with " f"{type(e)} : {e}",
                flush=True,
            )
            raise

    def _make_device(self) -> Device:
        """Create the device object which will determine on what hardware (CPU/GPU) each part of the simulation will
        be executed."""
        if self.config.desired_device in ["cuda", "cupy", "omp"]:
            try:
                cuda_devices = self.config.cuda_devices
            except AttributeError:
                cuda_devices = []

            lib = "omp" if self.config.desired_device == "omp" else "cuda"
            try:
                device = CudaDevice(self.comm, compiled_lib=lib, device_list=cuda_devices)
            except ValueError:
                device = None
                if self.comm.rank == 0:
                    print("Switching to CPU", flush=True)

            if device is None:
                device = CpuDevice(comm=self.comm)
        elif self.config.desired_device == "torch":
            device = PytorchDevice(comm=self.comm)
        else:
            device = CpuDevice(comm=self.comm)

        return device

    def _create_geometry(self) -> Geometry:
        """Create the appropriate geometry for the given problem"""

        if self.config.grid_file != "":
            self.process_topo = ProcessTopology(self.device, comm_in=self.comm)
            return CubedSphere2D(
                self.num_elements_horizontal,
                self.num_solpts,
                self.total_num_elements_horizontal,
                self.lambda0,
                self.phi0,
                self.alpha0,
                self.process_topo,
            )

        if self.config.grid_type == "cubed_sphere":
            self.process_topo = ProcessTopology(self.device, comm_in=self.comm)
            if self.config.equations == "shallow_water":
                return CubedSphere2D(
                    self.num_elements_horizontal,
                    self.num_solpts,
                    self.total_num_elements_horizontal,
                    self.lambda0,
                    self.phi0,
                    self.alpha0,
                    self.process_topo,
                )
            elif self.config.equations == "euler":
                cube_sphere = CubedSphere3D(
                    self.num_elements_horizontal,
                    self.config.num_elements_vertical,
                    self.num_solpts,
                    self.total_num_elements_horizontal,
                    self.lambda0,
                    self.phi0,
                    self.alpha0,
                    self.config.ztop,
                    self.process_topo,
                    self.config,
                )

                if self.config.enable_schar_mountain:
                    schar_mountain = ScharMountainHook(self.config, cube_sphere)
                    self.step_hooks[ScharMountainHook] = schar_mountain
                return cube_sphere

        if self.config.grid_type == "cartesian2d":
            return Cartesian2D(
                (self.config.x0, self.config.x1),
                (self.config.z0, self.config.z1),
                self.num_elements_horizontal,
                self.config.num_elements_vertical,
                self.num_solpts,
                self.total_num_elements_horizontal,
                self.device,
            )

        raise ValueError(f"Invalid grid type/process_topo: {self.config.grid_type}, {self.process_topo}")

    def _create_output_manager(self) -> OutputManager:
        if self.comm.rank == 0:
            print(
                "output_format =",
                self.config.output_format,
                flush=True,
            )

        if isinstance(self.geometry, Cartesian2D):
            return OutputCartesian(self.config, self.geometry, self.operators_real, self.device)
        elif isinstance(self.geometry, CubedSphere):
            if self.config.output_format == "netcdf":
                return OutputCubesphereNetcdf(
                    self.config,
                    self.geometry,
                    self.operators_real,
                    self.device,
                    self.metric,
                    self.topography,
                    self.process_topo,
                )
            elif self.config.output_format == "fst":
                return OutputCubesphereFst(
                    self.config,
                    self.geometry,
                    self.operators_real,
                    self.device,
                    self.metric,
                    self.topography,
                    self.process_topo,
                )
            elif self.config.output_format == "zarr":
                return OutputCubesphereZarr(
                    self.config,
                    self.geometry,
                    self.operators_real,
                    self.device,
                    self.metric,
                    self.topography,
                    self.process_topo,
                )

        raise ValueError(f"Unrecognized geometry type {type(self.geometry)}")

    def _adjust_num_elements(self):
        """Adjust number of horizontal elements in the parameters so that it corresponds to the
        number *per processor*."""
        if self.comm.size not in self.allowed_pe_counts:
            raise ValueError(
                f"Invalid number of processors for this particular "
                f"problem size ({self.num_elements_horizontal} elements per side). "
                f"\nAllowed counts are {self.allowed_pe_counts}"
            )

        self.total_num_elements_horizontal = self.num_elements_horizontal
        if self.comm.size > 1:
            num_pe_per_tile = self.comm.size // 6
            num_pe_per_line = int(np.sqrt(num_pe_per_tile))
            self.num_elements_horizontal = self.total_num_elements_horizontal // num_pe_per_line
            if self.comm.rank == 0:
                if self.total_num_elements_horizontal != self.num_elements_horizontal:
                    print(
                        f"Adjusting horizontal number of elements from {self.total_num_elements_horizontal} "
                        f"(total) to {self.num_elements_horizontal} (per PE)"
                    )
                print(f"allowed_pe_counts = {self.allowed_pe_counts}", flush=True)
