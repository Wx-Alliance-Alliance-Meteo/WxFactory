import os
import shutil
import tempfile

import numpy as np
import torch
import xarray as xr
import zarr
from mpi_test import MpiTestCase

from wx_factory.context import Context
from wx_factory.geometry import (
    Cartesian3D,
    CubedSphere2D,
    CubedSphere3D,
    DFROperators,
    GeometryContext,
    Metric2D,
    Metric3DTopo,
    resolve_geometry,
)
from wx_factory.output import InputManager
from wx_factory.output.registry import OutputContext, resolve_output
from wx_factory.step_hooks import ScharMountainHook
from wx_factory.step_hooks.registry import (
    PHASE_GEOMETRY,
    StepHookContext,
    resolve_step_hooks,
)
from wx_factory.wx_mpi import Conditional, SingleProcess


class CompareZarrToNcTestCase(MpiTestCase):
    def __init__(self, num_procs, methodName="runTest", optional=False):
        super().__init__(num_procs, methodName, optional)
        self.state_dir = "tests/data/unit/states_for_ouput_managers_tests"
        self.metric = None
        self.topography = None
        self.config = None
        self.context = None
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

        self.context = self._make_context(self.context)

        if hasattr(self.config, "precision"):
            if self.config.precision == "mixed":
                self.context.real_dtype = torch.float32
                self.context.complex_dtype = torch.complex64
            else:
                self.context.real_dtype = torch.float64
                self.context.complex_dtype = torch.complex128
        else:
            self.context.real_dtype = torch.float64
            self.context.complex_dtype = torch.complex128

        self.geometry = resolve_geometry(GeometryContext.from_simulation(self))

        self.process_topo = getattr(self.geometry, "process_topology", None)

        self.step_hooks.update(
            resolve_step_hooks(StepHookContext(config=self.config, geometry=self.geometry), phase=PHASE_GEOMETRY)
        )

        self.operators_real = DFROperators(self.geometry, self.context)

        if self.config.equations == "euler" and isinstance(self.geometry, Cartesian3D):
            self.metric = Metric3DTopo(self.geometry, self.operators)
            self.metric.build_metric()

        elif self.config.equations == "euler" and isinstance(self.geometry, CubedSphere3D):
            self.metric = Metric3DTopo(self.geometry, self.operators_real)
            if self.config.enable_schar_mountain:
                self.step_hooks[ScharMountainHook].metric = self.metric
                self.step_hooks[ScharMountainHook].apply(1 if self.config.schar_mountain_step == 0 else 0)
            self.metric.build_metric()

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
            # self.geometry.z_levels = self.Q.shape[1]
            self.geometry.z_levels = list(range(self.Q.shape[1]))
        else:
            self.geometry.z_levels = [0]

        self.output = resolve_output(
            OutputContext(
                config=self.config,
                context=self.context,
                geometry=self.geometry,
                operators=self.operators_real,
                metric=self.metric,
                topography=self.topography,
                ptopo=self.process_topo,
            )
        )
        self.output.step(self.Q, 0)
        self.output.finalize(0.0)
        self.comm.Barrier()

        return self.config, tmp_output_dir

    def _compare_outputs(self, nc_file, zarr_store):
        ds_nc = xr.open_dataset(nc_file)
        ds_zarr = zarr.open_group(zarr_store, mode="r")

        nc_vars = set(ds_nc.variables)
        zarr_vars = set(ds_zarr.array_keys())

        self.assertSetEqual(
            nc_vars,
            zarr_vars,
            (
                f"NetCDF and Zarr variable lists differ. "
                f"Only in NetCDF = {nc_vars - zarr_vars}, "
                f"Only in Zarr = {zarr_vars - nc_vars}"
            ),
        )

        try:
            for variable_name in ds_nc.variables:
                self.assertIn(
                    variable_name,
                    zarr_vars,
                    f"Variable '{variable_name}' missing from Zarr",
                )

                nc_values = ds_nc[variable_name].values
                zarr_values = ds_zarr[variable_name][:]

                self.assertEqual(
                    nc_values.shape,
                    zarr_values.shape,
                    (
                        f"Shape mismatch for variable '{variable_name}'. "
                        f"NetCDF={nc_values.shape}, "
                        f"Zarr={zarr_values.shape}"
                    ),
                )

                if not np.array_equal(nc_values, zarr_values):
                    diff = np.abs(nc_values - zarr_values)

                    mismatch_locations = np.argwhere(nc_values != zarr_values)

                    first_idx = tuple(mismatch_locations[0])

                    self.fail(
                        f"Variable '{variable_name}' differs.\n"
                        f"Number of mismatches : {len(mismatch_locations)}\n"
                        f"Maximum difference : {np.max(diff):.16e}\n"
                        f"First mismatch idx : {first_idx}\n"
                        f"NetCDF value       : {nc_values[first_idx]}\n"
                        f"Zarr value         : {zarr_values[first_idx]}\n"
                        f"Difference         : {diff[first_idx]}"
                    )

        finally:
            ds_nc.close()

    def test_compare_zarr_to_nc(self):
        try:
            pairs = self._find_pairs()
            if len(pairs) == 0:
                self.skipTest(f"No state-vector pairs found in {self.state_dir}")

            for nc_state_file, zarr_state_file in pairs:
                config_nc, tmp_nc_dir = self._generate_output(nc_state_file)
                config_zarr, tmp_zarr_dir = self._generate_output(zarr_state_file)

                if self.comm.rank == 0:
                    try:
                        nc_file = f"{tmp_nc_dir}/{config_nc.base_output_file}.nc"

                        zarr_store = f"{tmp_zarr_dir}/{config_zarr.base_output_file}.zarr"

                        if not os.path.exists(nc_file):
                            self.fail(f"Expected NetCDF output not found: {nc_file}")

                        if not os.path.exists(zarr_store):
                            self.fail(f"Expected Zarr output not found: {zarr_store}")
                        self._compare_outputs(
                            nc_file,
                            zarr_store,
                        )

                    finally:
                        shutil.rmtree(
                            tmp_nc_dir,
                            ignore_errors=True,
                        )

                        shutil.rmtree(
                            tmp_zarr_dir,
                            ignore_errors=True,
                        )
        except Exception as e:
            print(
                f"Rank {self.comm.rank} failed with {type(e)} : {e}",
                flush=True,
            )
            raise

    def _make_context(self, context: Context | None) -> Context:
        """Create the context object which will determine on what hardware (CPU/GPU) each part of the simulation will
        be executed."""
        if context is not None:
            self.comm = context.comm
            return context
        return Context(comm=self.comm, device_type=self.config.pytorch_device)

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
