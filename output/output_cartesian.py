import os
import numpy
import math

from common.definitions     import idx_2d_rho       as RHO,           \
                                   idx_2d_rho_w     as RHO_W,         \
                                   idx_2d_rho_theta as RHO_THETA
from common.graphx          import image_field
from common.configuration import Configuration
from geometry               import Geometry

def output_step(Q: numpy.ndarray, geom: Geometry, param: Configuration, filename: str) -> None:

   # TODO : hackathon SG
   nb_equations = Q.shape[0]
   Q_cartesian = numpy.empty((nb_equations, param.nb_elements_vertical*geom.nbsolpts, param.nb_elements_horizontal*geom.nbsolpts))

   for v in range(nb_equations):
      e = 0
      for ek in range(param.nb_elements_vertical):
         for ei in range(param.nb_elements_horizontal):
            start_i = ei * geom.nbsolpts 
            end_i = (ei + 1) * geom.nbsolpts
            start_k = ek * geom.nbsolpts 
            end_k = (ek + 1) * geom.nbsolpts
            Q_cartesian[v, start_k:end_k, start_i:end_i] = Q[v, e, :].reshape((geom.nbsolpts, geom.nbsolpts))
            e += 1

   if param.case_number == 0:
      image_field(geom, (Q_cartesian[RHO_W,:,:]), filename, -1, 1, 25, label='w (m/s)', colormap='bwr')
   elif param.case_number <= 2:
      image_field(geom, (Q_cartesian[RHO_THETA,:,:] / Q_cartesian[RHO,:,:]), filename, 303.1, 303.7, 7)
   elif param.case_number == 3:
      image_field(geom, (Q_cartesian[RHO_THETA,:,:] / Q_cartesian[RHO,:,:]), filename, 303., 303.7, 8)
   elif param.case_number == 4:
      image_field(geom, (Q_cartesian[RHO_THETA,:,:] / Q_cartesian[RHO,:,:]), filename, 290., 300., 10)
