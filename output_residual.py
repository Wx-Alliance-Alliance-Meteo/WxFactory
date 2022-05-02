from gef_mpi import GLOBAL_COMM

output_filename = 'test_result.txt'
output_param = None
is_writer = False

def prepare_output(param):
   
   global output_param
   output_param = param

   global is_writer
   if GLOBAL_COMM().rank == 0: is_writer = True
   if not is_writer: return

   try:
      f = open(output_filename)
      file_exists = True
      f.close()
   except:
      file_exists = False

   with open(output_filename, 'a+') as output_file:
      if not file_exists:
         output_file.write('# order | num_elements | dt | linear solver | precond | precond_interp | precond tol | coarsest MG order | MG smoothe only | # pre smoothe | # post smoothe | CFL # ::: FGMRES #it | FGMRES time | conv. flag \n')

def write_output(num_iter, time, flag, residuals):
   if not is_writer: return
   p = output_param
   with open(output_filename, 'a+') as output_file:
      # Params
      output_file.write(f'{p.nbsolpts} {p.nb_elements_horizontal:3d} {int(p.dt):5d} {p.linear_solver[:10]:10s} '
                        f'{p.preconditioner[:8]:8s} {p.dg_to_fv_interp[:8]:8s} {p.precond_tolerance:9.1e} '
                        f'{p.coarsest_mg_order:3d} {p.mg_smoothe_only} '
                        f'{p.num_pre_smoothe:3d} {p.num_post_smoothe:3d} {p.pseudo_cfl:7.3f} ::: ')

      # Sim results
      output_file.write(f'{num_iter:5d} {time:7.1f} ')
      output_file.write(f'{flag:2d} ')
      output_file.write(f'::: {" ".join(f"{r[0]:.2e}/{r[1]:.2e}/{r[2]}" for r in residuals)} ')

      output_file.write('\n')
