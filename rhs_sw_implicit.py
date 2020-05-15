from rhs_sw import rhs_sw
from rhs_sw_explicit import rhs_sw_explicit

def rhs_sw_implicit(Q, geom, mtrx, metric, topo, comm_dist_graph, nbsolpts, nb_elements_horiz, α, case_number):

   full_rhs = rhs_sw(Q, geom, mtrx, metric, topo, comm_dist_graph, nbsolpts, nb_elements_horiz, α, case_number)

   explicit_rhs = rhs_sw_explicit(Q, geom, mtrx, metric, topo, comm_dist_graph, nbsolpts, nb_elements_horiz, α, case_number)

   return full_rhs - explicit_rhs
