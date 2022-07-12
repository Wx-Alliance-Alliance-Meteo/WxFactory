import sqlite3
from gef_mpi import GLOBAL_COMM

class OutputManager:
   def __init__(self) -> None:
      # self.filename  = 'test_result.txt'
      self.db        = 'test_result.db'
      self.param     = None
      self.is_writer = False

      self.run_id    = -1
      self.step_id   = 0

      self.db_connection = sqlite3.connect(self.db)
      self.db_cursor     = self.db_connection.cursor()

   def prepare_output(self, param):
      self.param = param

      if GLOBAL_COMM().rank == 0: self.is_writer = True
      if not self.is_writer: return

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
               linear_solver     varchar(64),
               precond           varchar(64),
               precond_interp    varchar(64),
               precond_tol       float,
               kiops_dt_factor   float,
               num_mg_levels     int,
               mg_smoothe_only   bool,
               num_pre_smoothe   int,
               num_post_smoothe  int,
               num_solver_it     int,
               solver_time       float,
               solver_flag       int
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

      else:
         print(f'Table exists!')

   def write_output(self, num_iter, time, flag, residuals):
      if not self.is_writer: return
      p = self.param

      self.db_cursor.execute('''
         insert into results_param
         (run_id, step_id, dg_order, num_elem_h, num_elem_v, dt, linear_solver,
         precond, precond_interp, precond_tol,
         kiops_dt_factor,
         num_mg_levels, mg_smoothe_only, num_pre_smoothe, num_post_smoothe,
         num_solver_it, solver_time, solver_flag)
         values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         returning results_param.entry_id;''',
         [self.run_id, self.step_id, p.nbsolpts, p.nb_elements_horizontal, p.nb_elements_vertical, p.dt, p.linear_solver,
          p.preconditioner, p.dg_to_fv_interp, p.precond_tolerance,
          p.kiops_dt_factor, p.num_mg_levels,
          p.mg_smoothe_only, p.num_pre_smoothe, p.num_post_smoothe,
          num_iter, time, flag])

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

def prepare_output(param):
   output_mgr.prepare_output(param)

def write_output(num_iter, time, flag, residuals):
   output_mgr.write_output(num_iter, time, flag, residuals)
