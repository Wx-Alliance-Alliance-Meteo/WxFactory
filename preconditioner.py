from copy import copy

from cubed_sphere import cubed_sphere
from matrices     import DFR_operators
from metric       import Metric
from initialize   import initialize


def make_preconditioner_data(param, nb_sol_pts, cube_face):

   geom = cubed_sphere(param.nb_elements, nb_sol_pts, param.λ0, param.ϕ0, param.α0, cube_face)
   mtrx = DFR_operators(geom)
   metric = Metric(geom)

   param_small = copy(param)
   param_small.nbsolpts = nb_sol_pts
   _, topo = initialize(geom, metric, mtrx, param_small)

   return geom, mtrx, metric, topo


