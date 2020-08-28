import numpy

def vorticity(u1_contra, u2_contra, geom, metric, mtrx, param):
   
   u1_cov = metric.H_cov_11 * u1_contra + metric.H_cov_12 * u2_contra
   u2_cov = metric.H_cov_21 * u1_contra + metric.H_cov_22 * u2_contra

   du1dx2 = numpy.zeros_like(u1_contra)
   du2dx1 = numpy.zeros_like(u2_contra)

   for elem in range(param.nb_elements):
      epais = elem * param.nbsolpts + numpy.arange(param.nbsolpts)

      # --- Direction x1

      du2dx1[:,epais] = ( u2_cov[:,epais] @ mtrx.diff_tr ) * 2.0 / geom.Δx1

      # --- Direction x2

      du1dx2[epais,:] = ( mtrx.diff @ u1_cov[epais,:] ) * 2.0 / geom.Δx2

   vort = metric.inv_sqrtG * ( du2dx1 - du1dx2 )

   return vort
