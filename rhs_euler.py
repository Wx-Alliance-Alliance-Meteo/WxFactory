import numpy

from definitions import idx_rho_u1, idx_rho_u2, idx_rho_u3, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd
from rhs_euler_hori import * # TODO : mettre ensemble
from rhs_euler_vert import * # TODO : mettre ensemble

def rhs_euler(Q, geom, mtrx, metric, topo, ptopo, nb_solpts: int, nb_elements_horiz: int, nb_elements_vert: int, case_number: int, filter_rhs: bool = False):

   datatype = Q.dtype
   nb_equations = 5
   nb_interfaces_horiz = nb_elements_horiz + 1
   nb_total_sol_pt_horiz = nb_elements_horiz * nb_solpts
   nb_total_sol_pt_vert = nb_elements_vert * nb_solpts

   # Result
   rhs_H = numpy.zeros_like(Q)
   rhs_V = numpy.zeros_like(Q) # TODO
   
   for lvl in range(nb_elements_vert*nb_solpts):
      rhs_H[:,lvl,:,:] = rhs_euler_hori(Q[:,lvl,:,:], geom, mtrx, metric, topo, ptopo, nb_solpts, nb_elements_horiz, case_number, filter_rhs)

   for i in range(nb_elements_horiz*nb_solpts):
      for j in range(nb_elements_horiz*nb_solpts):
         rhs_V[:,:,i,j] = rhs_euler_vert(Q[:,:,i,j], metric.sqrtG[i,j], mtrx, nb_solpts, nb_elements_vert, case_number, filter_rhs) #, geom, mtrx, metric, topo, ptopo, , )

   return rhs_H + rhs_V