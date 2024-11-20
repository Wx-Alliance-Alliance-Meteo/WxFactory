import os
from typing import Callable, Optional, Union

from mpi4py import MPI
import numpy

from common.configuration import Configuration
from geometry import Cartesian2D, CubedSphere, Geometry, Metric, Metric3DTopo, DFROperators
from init.initialize import Topo
from output.blockstats import blockstats_cart, blockstats_cs
from output.solver_stats import SolverStatsOutput
from output.state import save_state
from precondition.multigrid import Multigrid
from solvers import SolverInfo


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
        metric: Optional[Union[Metric, Metric3DTopo]] = None,
        operators: Optional[DFROperators] = None,
        topo: Optional[Topo] = None,
    ) -> None:

        self.param = param
        self.geometry = geometry
        self.metric = metric
        self.operators = operators
        self.topo = topo

        if MPI.COMM_WORLD.rank == 0:
            try:
                os.makedirs(os.path.abspath(param.output_dir), exist_ok=True)
            except PermissionError:
                new_name = "results"
                print(
                    f'WARNING: Unable to create directory {param.output_dir} for output. Will use "./{new_name}" instead'
                )
                param.output_dir = new_name
                os.makedirs(os.path.abspath(param.output_dir), exist_ok=True)

        MPI.COMM_WORLD.bcast(param.output_dir, root=0)

        self.final_function = lambda x=None: None
        self.blockstat_function = lambda Q, step_id: None

        if self.param.store_solver_stats > 0:
            self.solver_stats_output = SolverStatsOutput(param)

        if param.output_freq > 0:
            if self.geometry.grid_type == "cubed_sphere":
                from output.output_cubesphere import output_init, output_netcdf, output_finalize

                output_init(self.geometry, self.param)
                self.step_function = lambda Q, step_id: output_netcdf(
                    Q, self.geometry, self.metric, self.operators, self.topo, step_id, self.param
                )
                self.final_function = output_finalize
            elif self.geometry.grid_type == "cartesian2d":
                from output.output_cartesian import output_step

                self.output_file_name = (
                    lambda step_id: f"{self.param.output_dir}/bubble_{self.param.case_number}_{step_id:08d}"
                )
                self.step_function = lambda Q, step_id: output_step(
                    Q, self.geometry, self.param, self.output_file_name(step_id)
                )

        if param.stat_freq > 0:
            if isinstance(self.geometry, CubedSphere):
                if self.param.equations == "shallow_water":
                    if self.topo is None:
                        raise ValueError(f"Need a topo for this!")
                    self.blockstat_function = lambda Q, step_id: blockstats_cs(
                        Q, self.geometry, self.topo, self.metric, self.operators, self.param, step_id
                    )
                else:
                    if MPI.COMM_WORLD.rank == 0:
                        print(f"WARNING: Blockstat only implemented for Shallow Water equations")
            elif isinstance(self.geometry, Cartesian2D):
                self.blockstat_function = lambda Q, step_id: blockstats_cart(Q, self.geometry, step_id)

        # Choose a file name hash based on a certain set of parameters:
        state_params = (
            param.dt,
            param.nb_elements_horizontal,
            param.nb_elements_vertical,
            param.nbsolpts,
            MPI.COMM_WORLD.size,
        )
        self.config_hash = state_params.__hash__() & 0xFFFFFFFFFFFF

    def state_file_name(self, step_id: int) -> str:
        """Return the name of the file where to save the state vector for the current problem, for the given timestep."""
        base_name = f"state_vector_{self.config_hash:012x}_{MPI.COMM_WORLD.rank:03d}"
        return f"{self.param.output_dir}/{base_name}.{step_id:08d}.npy"

    def step(self, Q: numpy.ndarray, step_id: int) -> None:
        """Output the result of the latest timestep."""
        if self.param.output_freq > 0:
            if step_id % self.param.output_freq == 0:
                if MPI.COMM_WORLD.rank == 0:
                    print(f"=> Writing dynamic output for step {step_id}")
                self.step_function(Q, step_id)

        if self.param.save_state_freq > 0:
            if step_id % self.param.save_state_freq == 0:
                save_state(Q, self.param, self.state_file_name(step_id))

        if self.param.stat_freq > 0:
            if step_id % self.param.stat_freq == 0:
                self.blockstat_function(Q, step_id)

    def store_solver_stats(
        self,
        total_time: float,
        simulation_time: float,
        dt: float,
        solver_info: SolverInfo,
        precond: Optional[Multigrid],
    ):
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
        Perform any necessary operation to properly finish outputting
        """
        if self.param.store_total_time:
            if MPI.COMM_WORLD.rank == 0:
                size = MPI.COMM_WORLD.size
                method = str(self.param.time_integrator)
                methodOrtho = str(self.param.exponential_solver)
                caseNum = str(self.param.case_number)
                totaltime_name = f"{self.param.output_dir}/runtime_{methodOrtho}_n{size}_{method}_c{caseNum}.txt"
                with open(totaltime_name, "a") as gg:
                    gg.write(f"{total_time}\n")

        if self.param.output_freq > 0:
            self.final_function()
