
import mpi4py.MPI

from typing  import Callable, Optional, Union
import numpy

from common.program_options import Configuration
from geometry.geometry      import Geometry
from geometry.metric        import Metric, Metric_3d_topo
from geometry.matrices      import DFR_operators
from init.initialize        import Topo
from output.blockstats      import blockstats
from output.solver_stats    import prepare_solver_stats

class OutputManager:
   """
   Class that uniformizes different output methods
   """
   final_function: Callable[[], None]
   output_file_name: Callable[[int], str]
   def __init__(self, \
                param: Configuration, \
                geometry: Geometry,     \
                metric: Optional[Union[Metric, Metric_3d_topo]] = None, \
                operators: Optional[DFR_operators] = None,              \
                topo: Optional[Topo] = None) -> None:

      self.param     = param
      self.geometry  = geometry
      self.metric    = metric
      self.operators = operators
      self.topo      = topo

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
            from output.output_cartesian import output_init, output_step
            output_init(self.param)

            self.output_file_name = lambda step_id: \
               f'{self.param.output_dir}/bubble_{self.param.case_number}_{step_id:07d}'
            self.step_function = lambda Q, step_id: output_step(Q, self.geometry, self.param, self.output_file_name(step_id))

      if param.store_solver_stats > 0:
         prepare_solver_stats(param)

      if param.stat_freq > 0 and self.geometry.grid_type == 'cubed_sphere':
         self.blockstat_function = lambda Q, step_id: \
            blockstats(Q, self.geometry, self.topo, self.metric, self.operators, self.param, step_id)
      
   def state_file_name(self, step_id: int) -> str:
      return f'{self.param.output_dir}/state_vector_{mpi4py.MPI.COMM_WORLD.rank:03d}.{step_id:07d}.npy'

   def step(self, Q: numpy.ndarray, step_id: int) -> None:
      """
      Output the result of the latest timestep
      """
      if self.param.output_freq > 0:
         if step_id % self.param.output_freq == 0:
            if mpi4py.MPI.COMM_WORLD.rank == 0: print(f'=> Writing dynamic output for step {step_id}')
            self.step_function(Q, step_id)

      if self.param.save_state_freq > 0:
         if step_id % self.param.save_state_freq == 0:
            numpy.save(self.state_file_name(step_id), Q)

      if self.param.stat_freq > 0:
         if step_id % self.param.stat_freq == 0:
            self.blockstat_function(Q, step_id)

   def finalize(self) -> None:
      """
      Perform any necessary operation to properly finish outputting
      """
      if self.param.output_freq > 0:
         self.final_function()
