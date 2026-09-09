import glob
import os
from configparser import ConfigParser, NoOptionError, NoSectionError
from typing import TypeVar

import torch
from mpi4py import MPI
from torch import Tensor

import wx_factory.wx_mpi
from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.output import state
from wx_factory.simulation import Simulation


def get_rel_diff(a: Tensor, b: Tensor) -> float:
    num_var = a.shape[0]
    a_sizes = [torch.linalg.norm(a[i]).item() for i in range(num_var)]
    norms = [torch.linalg.norm(b[i] - a[i]).item() / a_sizes[i] for i in range(num_var)]
    return sum(norms) / len(norms)


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
        self.config_files = glob.glob(f"{self.config_dir_path}/config*.ini")
        # print(f"Config files: {self.config_files}")

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

            relative_diff = get_rel_diff(true_data, data)

            if self.comm.rank == 0:
                print(f"relative diff = {relative_diff:.2e}", flush=True)

            error_threshold = self.error_threshold
            if error_threshold < 0:
                num_steps = true_config.t_end // true_config.dt
                error_threshold = config.tolerance if num_steps <= 1 else config.tolerance * num_steps / 2

            self.assertLessEqual(
                relative_diff, error_threshold, f"The relative difference ({relative_diff:.2e}) is too big"
            )
