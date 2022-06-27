import sqlite3
from gef_mpi import GLOBAL_COMM

class OutputManager:
   def __init__(self) -> None:
      self.filename  = 'test_result.txt'
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

      try:
         f = open(self.filename)
         file_exists = True
         f.close()
      except:
         file_exists = False

      with open(self.filename, 'a+') as output_file:
         if not file_exists:
            output_file.write('# order | num_elements (horizontal) | num_elements (vertical) | dt | linear solver | precond | precond_interp | precond tol | coarsest MG order | MG smoothe only | # pre smoothe | # post smoothe | CFL # ::: FGMRES #it | FGMRES time | conv. flag \n')

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
               entry_id       integer PRIMARY KEY,
               run_id         int,
               step_id        int,
               dg_order       int,
               num_elem_h     int,
               num_elem_v     int,
               dt             int,
               linear_solver  varchar(64),
               precond        varchar(64),
               precond_interp varchar(64),
               precond_tol    float,
               coarsest_mg_order int,
               mg_smoothe_only   bool,
               num_pre_smoothe   int,
               num_post_smoothe  int,
               pseudo_cfl        float,
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
      with open(self.filename, 'a+') as output_file:
         # Params
         output_file.write(f'{p.nbsolpts} {p.nb_elements_horizontal:3d} {p.nb_elements_vertical:3d} {int(p.dt):5d} {p.linear_solver[:10]:10s} '
                           f'{p.preconditioner[:8]:8s} {p.dg_to_fv_interp[:8]:8s} {p.precond_tolerance:9.1e} '
                           f'{p.coarsest_mg_order:3d} {True if p.mg_smoothe_only == 1 else False} '
                           f'{p.num_pre_smoothe:3d} {p.num_post_smoothe:3d} {p.pseudo_cfl:8.5f} ::: ')

         # Sim results
         output_file.write(f'{num_iter:5d} {time:7.1f} ')
         output_file.write(f'{flag:2d} ')
         output_file.write(f'::: {" ".join(f"{r[0]:.2e}/{r[1]:.2e}/{r[2]}" for r in residuals)} ')

         output_file.write('\n')

      self.db_cursor.execute('''
         insert into results_param
         (run_id, step_id, dg_order, num_elem_h, num_elem_v, dt, linear_solver, precond, precond_interp, precond_tol,
         coarsest_mg_order, mg_smoothe_only, num_pre_smoothe, num_post_smoothe, pseudo_cfl,
         num_solver_it, solver_time, solver_flag)
         values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         returning results_param.entry_id;''',
         [self.run_id, self.step_id, p.nbsolpts, p.nb_elements_horizontal, p.nb_elements_vertical, p.dt, p.linear_solver,
          p.preconditioner, p.dg_to_fv_interp, p.precond_tolerance, p.coarsest_mg_order,
          p.mg_smoothe_only, p.num_pre_smoothe, p.num_post_smoothe, p.pseudo_cfl,
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
