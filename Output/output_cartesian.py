import os
import numpy

from Common.definitions         import idx_2d_rho       as RHO,           \
                                       idx_2d_rho_theta as RHO_THETA
from Common.graphx              import image_field
from Common.program_options     import Configuration
from Geometry.geometry          import Geometry

def output_init(param: Configuration):
   """
   Create output directory
   """
   os.makedirs(os.path.abspath(param.output_dir), exist_ok=True)

def output_step(Q: numpy.ndarray, geom: Geometry, param: Configuration, filename: str) -> None:
   if param.case_number <= 2:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 303.1, 303.7, 7)
   elif param.case_number == 3:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 303., 303.7, 8)
   elif param.case_number == 4:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 290., 300., 10)
