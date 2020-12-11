import numpy

import diagnostic

def fix_mass(Q, geom, topo, metric, mtrx, param, step):

   if param.case_number < 2 or param.mass_fixer is False:
      return

   if step == 0:
      global initial_mass
      initial_mass = diagnostic.global_integral(Q[0], mtrx, metric, param.nbsolpts, param.nb_elements) 
   else:
      actual_mass = diagnostic.global_integral(Q[0], mtrx, metric, param.nbsolpts, param.nb_elements) 
      fix_factor = initial_mass / actual_mass
      Q[0] *= fix_factor
      print('Mass fixer factor :', fix_factor)
