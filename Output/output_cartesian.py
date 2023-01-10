
from Common.definitions import idx_2d_rho       as RHO,           \
                               idx_2d_rho_theta as RHO_THETA
from Common.graphx      import image_field

state_file_name = lambda step_id: f'state_vector_{step_id:05d}.npy'
output_file_name = lambda step_id: f'out_{step_id:05d}'

def output_init(geom, param):
   """
   Define output file names
   """
   global state_file_name
   global output_file_name

   state_file_name = lambda step_id: f'{param.output_dir}/state_vector_{step_id:05d}.npy'
   output_file_name = lambda step_id: f'{param.output_dir}/bubble_{param.case_number}_{step_id:05d}'

def output_step(Q, geom, step, param):
   if param.case_number <= 2:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), output_file_name(step), 303.1, 303.7, 7)
   elif param.case_number == 3:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), output_file_name(step), 303., 303.7, 8)
   elif param.case_number == 4:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), output_file_name(step), 290., 300., 10)

def output_finalize():
   pass
