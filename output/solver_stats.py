from mpi4py import MPI

try:
   import sqlite3
   sqlite_available = True
except ModuleNotFoundError:
   sqlite_available = False
   print(f'No sqlite, won\'t be able to print solver stats')

from common.program_options import Configuration

class OutputManager:
   param: Configuration
   db_connection: sqlite3.Connection
   db_cursor: sqlite3.Cursor
   def __init__(self) -> None:
      self.db_name   = 'solver_stats.db'
      # self.param     = None
      self.is_writer = False

      self.run_id    = -1
      self.step_id   = 0

      self.db_connection = None
      self.db_cursor     = None

   def prepare_output(self, param: Configuration):
      self.param = param

      self.is_writer = MPI.COMM_WORLD.rank == 0
      if not (sqlite_available and self.is_writer): return

      self.db = f'{self.param.output_dir}/{self.db_name}'
      self.db_connection = sqlite3.connect(self.db)
      self.db_cursor     = self.db_connection.cursor()

      self.create_results_table()

   def create_results_table(self):
      """
      Create the results tables in the database, if they don't already exist
      """
      # First make sure the results table does not exist yet in the DB
      self.db_cursor.execute('''
                     SELECT name FROM sqlite_master WHERE type='table' AND name=?;
                     ''', ['results_param'])
               
      if len(self.db_cursor.fetchall()) == 0:
         self.db_cursor.execute('''
            CREATE TABLE results_param (
               entry_id          integer PRIMARY KEY,
               run_id            int,
               step_id           int,
               dg_order          int,
               num_elem_h        int,
               num_elem_v        int,
               dt                int,
               time_integrator   varchar(64),
               solver_tol        float,
               precond           varchar(64),
               precond_interp    varchar(64),
               precond_tol       float,
               mg_smoother       varchar(64),
               kiops_dt_factor   float,
               num_mg_levels     int,
               mg_solve_coarsest bool,
               num_pre_smoothe   int,
               num_post_smoothe  int,
               pseudo_cfl        float,
               num_solver_it     int,
               solver_time       float,
               solver_flag       int,
               smoother_radii    varchar(128)
            );
         ''')
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

   def write_output(self, num_iter, time, flag, residuals):
      if not (sqlite_available and self.is_writer): return
      p = self.param

      self.db_cursor.execute('''
         insert into results_param
         (run_id, step_id, dg_order, num_elem_h, num_elem_v, dt,
         time_integrator, solver_tol,
         precond, precond_interp, precond_tol, mg_smoother,
         kiops_dt_factor,
         num_mg_levels, mg_solve_coarsest, num_pre_smoothe, num_post_smoothe,
         pseudo_cfl,
         num_solver_it, solver_time, solver_flag, smoother_radii)
         values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         returning results_param.entry_id;''',
         [self.run_id, self.step_id, p.nbsolpts, p.nb_elements_horizontal, p.nb_elements_vertical, p.dt,
          p.time_integrator, p.tolerance,
          p.preconditioner, p.dg_to_fv_interp, p.precond_tolerance, p.mg_smoother,
          p.kiops_dt_factor, p.num_mg_levels,
          p.mg_solve_coarsest, p.num_pre_smoothe, p.num_post_smoothe,
          p.pseudo_cfl,
          num_iter, time, flag,
          str(p.exp_smoothe_spectral_radii)])

      if self.run_id < 0:
         self.run_id = self.db_cursor.fetchall()[0][0]
         self.db_cursor.execute('''
         update results_param
         set run_id = ?
         where entry_id = ?
         ''',
         [self.run_id, self.run_id])

      self.db_cursor.executemany('''
            insert into results_data values (?, ?, ?, ?, ?, ?);
         ''',
         [[self.run_id, self.step_id, i, r[0], r[1], r[2]] for i, r in enumerate(residuals)]
      )

      self.db_connection.commit()
      # print(f'RESULT_ID = {self.run_id}')
      self.step_id += 1

output_mgr = OutputManager()

def prepare_solver_stats(param):
   output_mgr.prepare_output(param)

def write_solver_stats(num_iter, time, flag, residuals):
   output_mgr.write_output(num_iter, time, flag, residuals)
