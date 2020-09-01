import numpy

from definitions import gravity

def relative_vorticity(u1_contra, u2_contra, geom, metric, mtrx, param):
   
   u1_dual = metric.H_cov_11 * u1_contra + metric.H_cov_12 * u2_contra
   u2_dual = metric.H_cov_21 * u1_contra + metric.H_cov_22 * u2_contra

   du1dx2 = numpy.zeros_like(u1_contra)
   du2dx1 = numpy.zeros_like(u2_contra)

   for elem in range(param.nb_elements):
      epais = elem * param.nbsolpts + numpy.arange(param.nbsolpts)

      # --- Direction x1

      du2dx1[:,epais] = ( u2_dual[:,epais] @ mtrx.diff_tr ) * 2.0 / geom.Δx1

      # --- Direction x2

      du1dx2[epais,:] = ( mtrx.diff @ u1_dual[epais,:] ) * 2.0 / geom.Δx2

   vort = metric.inv_sqrtG * ( du2dx1 - du1dx2 )

   return vort

def potential_vorticity(h, u1_contra, u2_contra, geom, metric, mtrx, param):

   rv = relative_vorticity(u1_contra, u2_contra, geom, metric, mtrx, param)
   
   return ( rv + metric.coriolis_f ) / h

def absolute_vorticity(u1_contra, u2_contra, geom, metric, mtrx, param):

   rv = relative_vorticity(u1_contra, u2_contra, geom, metric, mtrx, param)
   
   return rv + metric.coriolis_f

def total_energy(h, u1_contra, u2_contra, geom, topo, metric):
   u1_dual = metric.H_cov_11 * u1_contra + metric.H_cov_12 * u2_contra
   u2_dual = metric.H_cov_21 * u1_contra + metric.H_cov_22 * u2_contra

   # Kinetic energy
   kinetic = 0.5 * (u1_dual * u1_contra + u2_dual * u2_contra)

   # Potential energy
   potential = gravity * (h + topo.hsurf)

   # "Total" energy
   return kinetic + potential

def potential_enstrophy(h, u1_contra, u2_contra, geom, metric, mtrx, param):
   rv = relative_vorticity(u1_contra, u2_contra, geom, metric, mtrx, param)
   return 0.5 * (rv + metric.coriolis_f)**2 / h

