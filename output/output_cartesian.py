import os
import numpy
import pdb

from common.definitions     import idx_2d_rho       as RHO,            \
                                   idx_2d_rho_w     as RHO_W,          \
                                   idx_2d_rho_theta as RHO_THETA,      \
                                   heat_capacity_ratio as gamma,       \
                                   cvd, gravity, cpd, p0, Rd, idx_2d_rho_u
from common.graphx          import image_field
from common.configuration import Configuration
from geometry               import Geometry

def output_step(Q: numpy.ndarray, geom: Geometry, param: Configuration, filename: str) -> None:
   if param.case_number == 0:
      image_field(geom, (Q[RHO_W,:,:]), filename, -1, 1, 25, label='w (m/s)', colormap='bwr')
   elif param.case_number <= 2:
      e = Q[RHO_THETA,:,:] / Q[RHO,:,:]
      Theta =  (e/cvd) - (gamma-1)*gravity*geom.X3 / cpd
      # R =  (Q[RHO_THETA,:,:] / Q[RHO,:,:])
      image_field(geom, Theta, filename, numpy.min(Theta), numpy.max(Theta), 8)
   elif param.case_number == 3:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 303., 303.7, 8)
   elif param.case_number == 4:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 290., 300., 10)
   elif param.case_number == 555:
      image_field(geom, Q[RHO,:,:], filename, Q[RHO].min(), Q[RHO].max(), 10, label='$\\rho$', colormap='summer')
