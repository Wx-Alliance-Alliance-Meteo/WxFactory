import glob
import os
from configparser import ConfigParser

import torch
from mpi4py import MPI
from torch import Tensor

import wx_factory.wx_mpi
from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.output import state
from wx_factory.simulation import Simulation


def component_errors(reference: Tensor, actual: Tensor, comm) -> list[float]:
    """Global relative L2 error per equation; zero references use absolute L2."""
    if not comm.allreduce(bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()), op=MPI.LAND):
        raise ValueError("Non-finite integration state")
    errors = []
    for expected, result in zip(reference, actual):
        norm_squared = comm.allreduce(expected.double().square().sum().item(), op=MPI.SUM)
        error_squared = comm.allreduce((result.double() - expected.double()).square().sum().item(), op=MPI.SUM)
        errors.append((error_squared / norm_squared) ** 0.5 if norm_squared else error_squared**0.5)
    return errors


class StateIntegrationTestCases(MpiTestCase):
    config_dir_path: str
    num_process_required: int

    def __init__(self, config_dir_path: str, device_name: str = "cuda"):
        super().__init__(MPI.COMM_WORLD.size, "test_state", device_name=device_name)
        self.config_dir_path = config_dir_path
        self.num_process_required = 0

        if not os.path.exists(self.config_dir_path):
            self.fail(f"Could not find test case {self.config_dir_path}")

        self.num_process_required = 0
        self.error_threshold = -1.0

    def setUp(self):
        super().setUp()
        if self.num_process_required > 0:
            self.assertEqual(
                MPI.COMM_WORLD.size,
                self.num_process_required,
                f"We are using {MPI.COMM_WORLD.size} process(es), but the test requires {self.num_process_required}",
            )

        self.schema = load_default_schema()
        self.config_files = sorted(glob.glob(f"{self.config_dir_path}/config*.ini"))
        self.assertTrue(self.config_files, "No integration configurations found")

        # Skip if the test reads an FST topography file but rmn/georef are unavailable
        topo_field = next((f for f in self.schema.fields if f.name == "topography_file"), None)
        if topo_field is not None:
            for config_file in self.config_files:
                parser = ConfigParser()
                parser.read(config_file)
                if parser.has_option(topo_field.section, "topography_file") and parser.get(
                    topo_field.section, "topography_file"
                ):
                    try:
                        import georef  # noqa: F401
                        import rmn  # noqa: F401
                    except (ImportError, OSError) as e:
                        self.skipTest(f"rmn/georef not available, cannot read FST topography: {e}")
                    break

    def test_state(self):
        for config_file in self.config_files:
            config_content = wx_factory.wx_mpi.do_once(readfile, config_file)

            config = Configuration(config_content, self.schema)

            sim = Simulation(config, context=self.context)
            sim.run()

            state_vector_file = sim.output.state_file_name(sim.step_id)
            base_name = os.path.split(state_vector_file)[-1]
            true_state_vector_file: str = f"{self.config_dir_path}/{base_name}"

            [data, _] = state.load_state(state_vector_file, device=self.context.torch_device)
            [true_data, true_config] = state.load_state(true_state_vector_file, device=self.context.torch_device)

            self.assertEqual(
                true_data.shape,
                data.shape,
                f"Result shape {data.shape} is different from reference solution {true_data.shape}",
            )

            errors = component_errors(true_data, data, self.comm)

            if self.comm.rank == 0:
                print(f"L2 error by equation: {errors}", flush=True)

            error_threshold = self.error_threshold
            if error_threshold < 0:
                num_steps = true_config.t_end // true_config.dt
                error_threshold = config.tolerance if num_steps <= 1 else config.tolerance * num_steps / 2

            for equation, error in enumerate(errors):
                with self.subTest(config=config_file, equation=equation):
                    self.assertLessEqual(error, error_threshold)
