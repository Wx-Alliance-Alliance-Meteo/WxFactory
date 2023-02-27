import os
from typing  import Callable, List, Optional, Tuple, Union

from mpi4py import MPI
import numpy

from common.program_options import Configuration
from geometry               import Geometry, Metric, Metric3DTopo, DFROperators
from init.initialize        import Topo
from output.blockstats      import blockstats
from output.solver_stats    import SolverStatsOutput
from solvers                import SolverInfo

class OutputManager:
   """
   Class that uniformizes different output methods
   """
   final_function: Callable[[], None]
   output_file_name: Callable[[int], str]
   def __init__(self,
                param: Configuration,
                geometry: Geometry,
                metric: Optional[Union[Metric, Metric3DTopo]] = None,
                operators: Optional[DFROperators] = None,
                topo: Optional[Topo] = None) -> None:

      self.param     = param
      self.geometry  = geometry
      self.metric    = metric
      self.operators = operators
      self.topo      = topo

      os.makedirs(os.path.abspath(param.output_dir), exist_ok=True)

      self.solver_stats_output = SolverStatsOutput(param)

      self.final_function = lambda x=None: None
      self.blockstat_function = lambda Q, step_id: None

      if param.output_freq > 0:
         if self.geometry.grid_type == 'cubed_sphere':
            from output.output_cubesphere import output_init, output_netcdf, output_finalize
            output_init(self.geometry, self.param)
            self.step_function = lambda Q, step_id: \
               output_netcdf(Q, self.geometry, self.metric, self.operators, self.topo, step_id, self.param)
            self.final_function = output_finalize
         elif self.geometry.grid_type == 'cartesian2d':
            from output.output_cartesian import output_step

            self.output_file_name = lambda step_id: \
               f'{self.param.output_dir}/bubble_{self.param.case_number}_{step_id:05d}'
            self.step_function = lambda Q, step_id: \
               output_step(Q, self.geometry, self.param, self.output_file_name(step_id))

      if param.stat_freq > 0 and self.geometry.grid_type == 'cubed_sphere':
         if self.param.equations == 'shallow_water':
            if self.topo is None:
               raise ValueError(f'Need a topo for this!')
            self.blockstat_function = lambda Q, step_id: \
               blockstats(Q, self.geometry, self.topo, self.metric, self.operators, self.param, step_id)
         else:
            print(f'WARNING: Blockstat only implemented for Shallow Water equations')

      state_params = (param.dt, param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts)
      self.config_hash = state_params.__hash__() & 0xffffffffffff

   def state_file_name(self, step_id: int) -> str:
      """Return the name of the file where to save the state vector for the current problem, for the given timestep."""
      base_name = f'state_vector_{self.config_hash:012x}_{MPI.COMM_WORLD.rank:03d}'
      return f'{self.param.output_dir}/{base_name}.{step_id:05d}.npy'

   def step(self, Q: numpy.ndarray, step_id: int) -> None:
      """Output the result of the latest timestep."""
      if self.param.output_freq > 0:
         if step_id % self.param.output_freq == 0:
            if MPI.COMM_WORLD.rank == 0: print(f'=> Writing dynamic output for step {step_id}')
            self.step_function(Q, step_id)

      if self.param.save_state_freq > 0:
         if step_id % self.param.save_state_freq == 0:
            numpy.save(self.state_file_name(step_id), Q)

      if self.param.stat_freq > 0:
         if step_id % self.param.stat_freq == 0:
            self.blockstat_function(Q, step_id)

   def store_solver_stats(self, total_time: float, simulation_time: float, dt: float, solver_info: SolverInfo, has_precond: bool):
      if self.param.store_solver_stats > 0:
         self.solver_stats_output.write_output(
            total_time, simulation_time, dt, solver_info.total_num_it, solver_info.time, solver_info.flag,
            solver_info.iterations, has_precond)

   def finalize(self) -> None:
      """
      Perform any necessary operation to properly finish outputting
      """
      if self.param.output_freq > 0:
         self.final_function()
