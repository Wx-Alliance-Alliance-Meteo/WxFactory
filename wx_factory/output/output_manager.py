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
from .state import save_state, load_state


def _readable_time(seconds):
    if seconds == 0.0:
        return "0 s"
    elif seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    elif seconds > 1000:
        h_s = int(seconds)
        h = h_s // 3600
        rem = seconds - h_s
        m = rem // 60
        s = rem % 60
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{seconds:.2f}"


class OutputManager:
    """
    Class that uniformizes different output methods
    """

    final_function: Callable[[], None]
    output_file_name: Callable[[int], str]

    def __init__(
        self,
        config: Configuration,
        geometry: Geometry,
        operators: DFROperators,
        device: Device,
    ) -> None:

        self.config = config
        self.geometry = geometry
        self.operators = operators
        self.device = device

        self.output_dir = self.config.output_dir

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

        if self.config.store_solver_stats > 0:
            self.solver_stats_output = SolverStatsOutput(config)

        # Choose a file name hash based on a certain set of parameters:
        state_params = (
            config.dt,
            config.num_elements_horizontal,
            config.num_elements_vertical,
            config.num_solpts,
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

    def load_state_from_file(self, step_id, sh):
        starting_state, _ = load_state(self.state_file_name(step_id), schema=self.config.schema, device=self.device)
        if starting_state.shape != sh:
            raise ValueError(
                f"ERROR reading state vector from file for step {step_id}. "
                f"The shape is wrong! ({starting_state.shape}, should be {sh})"
            )
        Q = self.device.xp.asarray(starting_state)

        if MPI.COMM_WORLD.rank == 0:
            print(f"Starting simulation from step {step_id} (rather than 0)")
            if step_id * self.config.dt >= self.config.t_end:
                print(
                    f"WARNING: Won't run any steps, since we will stop at step "
                    f"{int(self.device.xp.ceil(self.config.t_end / self.config.dt))}"
                )

        return Q, step_id

    # def save_state(self, Q):

    def step(self, Q: NDArray, step_id: int) -> None:
        """Output the result of the latest timestep."""
        if self.config.output_freq > 0 and (step_id % self.config.output_freq) == 0:
            if MPI.COMM_WORLD.rank == 0:
                print(f"=> Writing dynamic output for step {step_id}")

            t0 = time()

            self.__write_result__(Q, step_id)

            self.total_write_time += time() - t0
            self.num_writes += 1

        if self.config.save_state_freq > 0 and (step_id % self.config.save_state_freq) == 0:
            t0 = time()
            save_state(Q, self.config, self.state_file_name(step_id))
            self.total_save_state_time += time() - t0
            self.num_save_states += 1

        if self.config.stat_freq > 0 and (step_id % self.config.stat_freq) == 0:
            t0 = time()
            self.__blockstats__(Q, step_id)
            self.total_blockstat_time += time() - t0
            self.num_blockstats += 1

    def _gather_field(self, field: NDArray) -> NDArray:
        return field

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
        if self.config.store_solver_stats > 0:
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
        if self.config.store_total_time:
            if MPI.COMM_WORLD.rank == 0:
                size = MPI.COMM_WORLD.size
                method = str(self.config.time_integrator)
                methodOrtho = str(self.config.exponential_solver)
                caseNum = str(self.config.case_number)
                totaltime_name = f"{self.output_dir}/runtime_{methodOrtho}_n{size}_{method}_c{caseNum}.txt"
                with open(totaltime_name, "a") as gg:
                    gg.write(f"{total_time}\n")

        if self.config.output_freq > 0:
            self.__finalize__()

        if MPI.COMM_WORLD.rank == 0:
            per_write = self.total_write_time / self.num_writes if self.num_writes > 0 else 0.0
            per_save = self.total_save_state_time / self.num_save_states if self.num_save_states > 0 else 0.0
            per_blockstat = self.total_blockstat_time / self.num_blockstats if self.num_blockstats > 0 else 0.0
            print(
                f"Output time:\n"
                f" - Write solution: {_readable_time(self.total_write_time)} "
                f"({_readable_time(per_write)}/step)\n"
                f" - Save state: {_readable_time(self.total_save_state_time)} "
                f"({_readable_time(per_save)}/step)\n"
                f" - Blockstats: {_readable_time(self.total_blockstat_time)} "
                f"({_readable_time(per_blockstat)}/step)"
            )

    def __finalize__(self):
        """Class-specific finalization"""
        # Not implemented by default
