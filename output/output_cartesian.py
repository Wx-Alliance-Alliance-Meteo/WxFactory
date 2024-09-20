import numpy

from common.definitions    import idx_2d_rho       as RHO,           \
                                  idx_2d_rho_w     as RHO_W,         \
                                  idx_2d_rho_theta as RHO_THETA
from common.graphx         import image_field
from common.configuration  import Configuration
from geometry              import Cartesian2D

def output_step(Q: numpy.ndarray, geom: Cartesian2D, param: Configuration, filename: str) -> None:

   Q_cartesian = geom.to_single_block(Q)

   if param.case_number == 0:
      image_field(geom, (Q_cartesian[RHO_W,:,:]), filename, -1, 1, 25, label='w (m/s)', colormap='bwr')
   elif param.case_number <= 2:
      image_field(geom, (Q_cartesian[RHO_THETA,:,:] / Q_cartesian[RHO,:,:]), filename, 303.1, 303.7, 7)
   elif param.case_number == 3:
      image_field(geom, (Q_cartesian[RHO_THETA,:,:] / Q_cartesian[RHO,:,:]), filename, 303., 303.7, 8)
   elif param.case_number == 4:
      image_field(geom, (Q_cartesian[RHO_THETA,:,:] / Q_cartesian[RHO,:,:]), filename, 290., 300., 10)
