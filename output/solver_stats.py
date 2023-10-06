from copy   import deepcopy
from typing import Any, List, Optional, Tuple

from mpi4py import MPI

try:
   import sqlite3
   sqlite_available = True
except ModuleNotFoundError:
   sqlite_available = False
   print(f'No sqlite, won\'t be able to print solver stats')

from common.program_options import Configuration
from precondition.multigrid import Multigrid

class Column:
   value: Any
   def __init__(self, col_type: str, col_value: Optional[Any] = None) -> None:
      self.type = col_type
      if col_value is not None:
         self.value = col_value

class ColumnSet:
   def __init__(self, param: Configuration) -> None:
      self.run_id            = Column('int', -1)
      self.step_id           = Column('int', 0)
      self.dg_order          = Column('int', param.nbsolpts)
      self.num_elem_h        = Column('int', param.nb_elements_horizontal_total)
      self.num_elem_v        = Column('int', param.nb_elements_vertical)
      self.initial_dt        = Column('int', param.dt)
      self.dt                = Column('int', param.dt)
      self.equations         = Column('varchar(64)', param.equations)
      self.case_number       = Column('int', param.case_number)
      self.grid_type         = Column('varchar(64)', param.grid_type)
      self.time_integrator   = Column('varchar(64)', param.time_integrator)
      self.solver_tol        = Column('float', param.tolerance)
      self.gmres_restart     = Column('int', param.gmres_restart)
      self.precond           = Column('varchar(64)', param.preconditioner)
      self.precond_interp    = Column('varchar(64)', param.dg_to_fv_interp)
      self.precond_tol       = Column('float', param.precond_tolerance)
      self.mg_smoother       = Column('varchar(64)', param.mg_smoother)
      self.kiops_dt_factor   = Column('float', param.kiops_dt_factor)
      self.num_mg_levels     = Column('int', param.num_mg_levels)
      self.mg_solve_coarsest = Column('bool', param.mg_solve_coarsest)
      self.num_pre_smoothe   = Column('int', param.num_pre_smoothe)
      self.num_post_smoothe  = Column('int', param.num_post_smoothe)
      self.pseudo_cfl        = Column('float', param.pseudo_cfl)
      self.simulation_time   = Column('float')
      self.total_solve_time  = Column('float')
      self.num_solver_it     = Column('int')
      self.solver_time       = Column('float')
      self.solver_flag       = Column('int')
      self.smoother_radii    = Column('varchar(128)', str(param.exp_smoothe_spectral_radii))

      self.exp_radius_0 = Column('float', 0)
      self.exp_radius_1 = Column('float', 0)
      self.exp_radius_2 = Column('float', 0)
      self.exp_radius_3 = Column('float', 0)
      self.exp_radius_4 = Column('float', 0)

      self.num_procs    = Column('int', MPI.COMM_WORLD.size)

      if param.grid_type == 'cartesian2d':
         self.x0   = Column('float', param.x0)
         self.x1   = Column('float', param.x1)
         self.z0   = Column('float', param.z0)
         self.z1   = Column('float', param.z1)
         self.ztop = Column('float', 0)
      elif param.grid_type == 'cubed_sphere':
         self.x0   = Column('float', 0)
         self.x1   = Column('float', 0)
         self.z0   = Column('float', 0)
         self.z1   = Column('float', 0)
         self.ztop = Column('float', param.ztop)


class SolverStatsOutput:
   """Contains necessary info to store solver stats into a SQL database"""

   def __init__(self, param: Configuration) -> None:
      """Connect to the DB file and create (if necessary) the tables. Only 1 PE will perform DB operations. """

      # Only 1 PE will connect to the DB and log solver stats
      self.is_writer = MPI.COMM_WORLD.rank == 0
      if not (sqlite_available and self.is_writer):
         if MPI.COMM_WORLD.allreduce(0) != 0:
            raise ValueError("Seems like init failed on root PE...")
         return

      self.param = _sanitize_params(param)
      self.columns = ColumnSet(self.param)
      self.param_table = 'results_param'

      self.columns.run_id.value  = -1
      self.columns.step_id.value = 0

      self.db_name       = self.param.solver_stats_file
      self.db            = f'{self.param.output_dir}/{self.db_name}'

      try:
         self.db_connection = sqlite3.connect(self.db)
         self.db_cursor     = self.db_connection.cursor()
         self.create_results_table()
         MPI.COMM_WORLD.allreduce(0)

      except sqlite3.OperationalError:
         # Signal failure to the other PEs
         MPI.COMM_WORLD.allreduce(1)
         raise

   def create_results_table(self):
      """Create the results tables in the database, if they don't already exist."""
      # First make sure the results table does not exist yet in the DB
      self.db_cursor.execute('''
                     SELECT name FROM sqlite_master WHERE type='table' AND name=?;
                     ''', [self.param_table])

      if len(self.db_cursor.fetchall()) == 0: # The table does not exist
         create_string = f'''
            CREATE TABLE {self.param_table} (
               entry_id          integer PRIMARY KEY,
               {', '.join([f'{name} {col.type}' for name, col in self.columns.__dict__.items()])}
            );
            '''
         self.db_cursor.execute(create_string)

         self.db_cursor.execute('''
            CREATE TABLE results_data (
               run_id    int,
               step_id   int,
               iteration int,
               residual  float(23),
               time      float(23),
               work      float(23)
            );
         ''')
         self.db_connection.commit()
      else: # The table exists, check its columns
         self.db_cursor.execute(f'PRAGMA table_info({self.param_table})')
         col_info = self.db_cursor.fetchall()
         col_names = [c[1] for c in col_info]
         cols = self.columns.__dict__
         if len(col_info) > len(cols) + 1:
            print(f'Woahhhh too many columns in the existing table!')
         elif len(cols) > len(col_info) - 1:
            for col_name, col in cols.items():
               if col_name not in col_names:
                  # Add that column to the table
                  print(f'Adding column {col_name} to solver params table')
                  default_value = 'none'
                  if col.type in ['int', 'float']: default_value = -1
                  add_string = f'''
                     ALTER TABLE {self.param_table}
                     ADD COLUMN {col_name} {col.type}
                     DEFAULT {default_value}
                  '''
                  print(f'Add string = {add_string}')
                  self.db_cursor.execute(add_string)


   def write_output(self,
                    total_time:float,
                    simulation_time: float,
                    dt: float,
                    num_iter: int,
                    local_time: float,
                    flag: int,
                    residuals: List[Tuple[float, float, float]],
                    precond: Optional[Multigrid]):
      try:
         self._exec_write_output(total_time, simulation_time, dt, num_iter, local_time, flag, residuals, precond)      
      except sqlite3.OperationalError as e:
         print(f'Got an SQL error. Not gonna store anything at this point.')
         print(str(e))


   def _exec_write_output(self,
                    total_time:float,
                    simulation_time: float,
                    dt: float,
                    num_iter: int,
                    local_time: float,
                    flag: int,
                    residuals: List[Tuple[float, float, float]],
                    precond: Optional[Multigrid]):

      if not (sqlite_available and self.is_writer): return

      self.columns.precond.value          = self.param.preconditioner if precond is not None else 'none'
      self.columns.dt.value               = dt
      self.columns.simulation_time.value  = simulation_time
      self.columns.total_solve_time.value = total_time
      self.columns.num_solver_it.value    = num_iter
      self.columns.solver_time.value      = local_time
      self.columns.solver_flag.value      = flag

      if len(self.param.exp_smoothe_spectral_radii) > 0 and precond is not None:
         if len(precond.spectral_radii) > 0: self.columns.exp_radius_0.value = precond.spectral_radii[0]
         if len(precond.spectral_radii) > 1: self.columns.exp_radius_1.value = precond.spectral_radii[1]
         if len(precond.spectral_radii) > 2: self.columns.exp_radius_2.value = precond.spectral_radii[2]
         if len(precond.spectral_radii) > 3: self.columns.exp_radius_3.value = precond.spectral_radii[3]
         if len(precond.spectral_radii) > 4: self.columns.exp_radius_4.value = precond.spectral_radii[4]

      insert_string = f'''
         insert into results_param ({', '.join([f'{name}' for name in self.columns.__dict__])})
         values ({', '.join(['?' for _ in self.columns.__dict__])})
         returning results_param.entry_id;'''
      insert_values = [col.value for _, col in self.columns.__dict__.items()]
      self.db_cursor.execute(insert_string, insert_values)

      if self.columns.run_id.value < 0:
         self.columns.run_id.value = self.db_cursor.fetchall()[0][0]
         self.db_cursor.execute('''
         update results_param
         set run_id = ?
         where entry_id = ?
         ''',
         [self.columns.run_id.value, self.columns.run_id.value])

      self.db_cursor.executemany('''
            insert into results_data values (?, ?, ?, ?, ?, ?);
         ''',
         [[self.columns.run_id.value, self.columns.step_id.value, i, r[0], r[1], r[2]] for i, r in enumerate(residuals)]
      )

      self.db_connection.commit()
      self.columns.step_id.value += 1

def _sanitize_params(params: Configuration) -> Configuration:
   new_p = deepcopy(params)

   if new_p.preconditioner == 'none':
      new_p.dg_to_fv_interp = 'none'
      new_p.precond_tolerance = 0.0
      new_p.mg_smoother = 'none'
      new_p.kiops_dt_factor = 0.0
      new_p.num_mg_levels = 0
      new_p.mg_solve_coarsest = True
      new_p.num_pre_smoothe = 0
      new_p.num_post_smoothe = 0
      new_p.pseudo_cfl = 0.0
      new_p.exp_smoothe_spectral_radii = []
      new_p.exp_smoothe_nb_iters = []
   elif new_p.preconditioner in ['p-mg', 'fv-mg']:
      if new_p.mg_smoother in ['kiops', 'exp']:
         new_p.pseudo_cfl = 0.0
      if new_p.mg_smoother in ['erk1', 'erk3', 'ark1', 'exp']:
         new_p.kiops_dt_factor = 0.0
      if new_p.mg_smoother in ['erk1', 'erk3', 'ark1', 'kiops']:
         new_p.exp_smoothe_spectral_radii = []
         new_p.exp_smoothe_nb_iters = []

   return new_p
