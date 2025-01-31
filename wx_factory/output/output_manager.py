import os
from time import time
from typing import Callable, Optional

from mpi4py import MPI
from numpy.typing import NDArray

from common.configuration import Configuration
from device import Device
from geometry import Geometry, DFROperators
from precondition.multigrid import Multigrid
from solvers import SolverInfo

from .solver_stats import SolverStatsOutput
from .state import save_state


class OutputManager:
    """
    Class that uniformizes different output methods
    """

    final_function: Callable[[], None]
    output_file_name: Callable[[int], str]

    def __init__(
        self,
        param: Configuration,
        geometry: Geometry,
        operators: DFROperators,
        device: Device,
    ) -> None:

        self.param = param
        self.geometry = geometry
        self.operators = operators
        self.device = device

        self.output_dir = self.param.output_dir

        if MPI.COMM_WORLD.rank == 0:
            try:
                os.makedirs(os.path.abspath(self.output_dir), exist_ok=True)
            except PermissionError:
                new_name = "results"
                print(
                    f"WARNING: Unable to create directory {self.output_dir} for output. "
                    f"Will use './{new_name}' instead"
                )
                self.output_dir = new_name
                os.makedirs(os.path.abspath(self.output_dir), exist_ok=True)

        MPI.COMM_WORLD.bcast(self.output_dir, root=0)

        if self.param.store_solver_stats > 0:
            self.solver_stats_output = SolverStatsOutput(param)

        # Choose a file name hash based on a certain set of parameters:
        state_params = (
            param.dt,
            param.num_elements_horizontal,
            param.num_elements_vertical,
            param.num_solpts,
            MPI.COMM_WORLD.size,
        )
        self.config_hash = state_params.__hash__() & 0xFFFFFFFFFFFF

        self.num_writes = 0
        self.num_save_states = 0
        self.num_blockstats = 0
        self.total_write_time = 0.0
        self.total_save_state_time = 0.0
        self.total_blockstat_time = 0.0

    def state_file_name(self, step_id: int) -> str:
        """Return the name of the file where to save the state vector for the current problem,
        for the given timestep."""
        base_name = f"state_vector_{self.config_hash:012x}_{MPI.COMM_WORLD.rank:06d}"
        return f"{self.output_dir}/{base_name}.{step_id:08d}.npy"

    def step(self, Q: NDArray, step_id: int) -> None:
        """Output the result of the latest timestep."""
        if self.param.output_freq > 0 and (step_id % self.param.output_freq) == 0:
            if MPI.COMM_WORLD.rank == 0:
                print(f"=> Writing dynamic output for step {step_id}")

            t0 = time()

            self.__write_result__(Q, step_id)

            self.total_write_time += time() - t0
            self.num_writes += 1

        if self.param.save_state_freq > 0 and (step_id % self.param.save_state_freq) == 0:
            t0 = time()
            save_state(Q, self.param, self.state_file_name(step_id))
            self.total_save_state_time += time() - t0
            self.num_save_states += 1

        if self.param.stat_freq > 0 and (step_id % self.param.stat_freq) == 0:
            t0 = time()
            self.__blockstats__(Q, step_id)
            self.total_blockstat_time += time() - t0
            self.num_blockstats += 1

    def __write_result__(self, Q: NDArray, step_id: int):
        """Class-specific write implementation."""
        # Not implemented by default

    def __blockstats__(self, Q: NDArray, step_id: int):
        """Class-specific blockstats implementation."""
        # Not implemented by default

    def store_solver_stats(
        self,
        total_time: float,
        simulation_time: float,
        dt: float,
        solver_info: SolverInfo,
        precond: Optional[Multigrid],
    ):
        """Store statistics for the current step into a database."""
        if self.param.store_solver_stats > 0:
            self.solver_stats_output.write_output(
                total_time,
                simulation_time,
                dt,
                solver_info.total_num_it,
                solver_info.time,
                solver_info.flag,
                solver_info.iterations,
                precond,
            )

    def finalize(self, total_time: float) -> None:
        """
        Perform any necessary operation to properly finish outputting.
        """
        if self.param.store_total_time:
            if MPI.COMM_WORLD.rank == 0:
                size = MPI.COMM_WORLD.size
                method = str(self.param.time_integrator)
                methodOrtho = str(self.param.exponential_solver)
                caseNum = str(self.param.case_number)
                totaltime_name = f"{self.output_dir}/runtime_{methodOrtho}_n{size}_{method}_c{caseNum}.txt"
                with open(totaltime_name, "a") as gg:
                    gg.write(f"{total_time}\n")

        if self.param.output_freq > 0:
            self.__finalize__()

        if MPI.COMM_WORLD.rank == 0:
            per_write = self.total_write_time / self.num_writes if self.num_writes > 0 else 0.0
            per_save = self.total_save_state_time / self.num_save_states if self.num_save_states > 0 else 0.0
            per_blockstat = self.total_blockstat_time / self.num_blockstats if self.num_blockstats > 0 else 0.0
            print(
                f"Output time:\n"
                f" - Write solution: {self.total_write_time:.3f}s "
                f"({per_write:.3f} s/step)\n"
                f" - Save state: {self.total_save_state_time:.3f}s "
                f"({per_save:.3f} s/step)\n"
                f" - Blockstats: {self.total_blockstat_time:.3f}s "
                f"({per_blockstat:.3f} s/step)"
            )

    def __finalize__(self):
        """Class-specific finalization"""
        # Not implemented by default
