from mpi4py import MPI
import numpy
import pdb


from common.definitions import gravity

def relative_vorticity(u1_contra, u2_contra, geom, metric, mtrx, param):

   u1_dual = metric.H_cov_11 * u1_contra + metric.H_cov_12 * u2_contra
   u2_dual = metric.H_cov_21 * u1_contra + metric.H_cov_22 * u2_contra

   du1dx2 = numpy.zeros_like(u1_contra)
   du2dx1 = numpy.zeros_like(u2_contra)

   for elem in range(param.nb_elements_horizontal):
      epais = elem * param.nbsolpts + numpy.arange(param.nbsolpts)

      # --- Direction x1

      du2dx1[:,epais] = u2_dual[:,epais] @ mtrx.diff_tr

      # --- Direction x2

      du1dx2[epais,:] = mtrx.diff @ u1_dual[epais,:]

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
   kinetic = 0.5 * h * (u1_dual * u1_contra + u2_dual * u2_contra)

   # Potential energy
   potential = 0.5 * gravity * ( (h + topo.hsurf)**2 - topo.hsurf**2 )

   # "Total" energy
   return kinetic + potential

def potential_enstrophy(h, u1_contra, u2_contra, geom, metric, mtrx, param):
   rv = relative_vorticity(u1_contra, u2_contra, geom, metric, mtrx, param)
   return (rv + metric.coriolis_f)**2 / (2 * h)

def global_integral(field, mtrx, metric, nbsolpts, nb_elements_horiz):
   local_sum = 0.
   for line in range(nb_elements_horiz):
      min_lin, max_lin = line * nbsolpts + numpy.array([0, nbsolpts])
      for column in range(nb_elements_horiz):
         min_col, max_col = column * nbsolpts + numpy.array([0, nbsolpts])
         local_sum += numpy.sum( field[min_lin:max_lin,min_col:max_col] * metric.sqrtG[min_lin:max_lin,min_col:max_col] * mtrx.quad_weights )
   return MPI.COMM_WORLD.allreduce(local_sum)


def global_integral_cartesian(field, mtrx, nbsolpts, nb_elements_horizontal, nb_elements_vertical, geom):
   local_sum = 0.
   for line in range(nb_elements_vertical):
      min_lin, max_lin = line * nbsolpts + numpy.array([0, nbsolpts])
      for column in range(nb_elements_horizontal):
         min_col, max_col = column * nbsolpts + numpy.array([0, nbsolpts])
         local_sum += numpy.sum( field[min_lin:max_lin,min_col:max_col] * mtrx.quad_weights * (geom.Δx1/2) * (geom.Δx3/2))
   return MPI.COMM_WORLD.allreduce(local_sum)


def global_integral_3d(field, geom, mtrx, metric, nbsolpts, nb_elements_horiz, nb_elements_vert):
    local_sum_z = 0.0

    for layer in range(nb_elements_vert):
        min_layer, max_layer = layer * nbsolpts + numpy.array([0, nbsolpts])
        counter = numpy.zeros((nbsolpts))
        for r in range(nbsolpts):
            local_sum = 0
            for line in range(nb_elements_horiz):
                  min_lin, max_lin = line * nbsolpts + numpy.array([0, nbsolpts])
                  for column in range(nb_elements_horiz):
                     min_col, max_col = column * nbsolpts + numpy.array([0, nbsolpts])
                     element_field = field[min_layer:max_layer,min_lin:max_lin,min_col:max_col]
                     element_sqrtG = metric.sqrtG[min_layer:max_layer,min_lin:max_lin,min_col:max_col]
                     local_sum += numpy.sum( element_field[r] * element_sqrtG[r] * mtrx.quad_weights )
            counter[r] = local_sum
        local_sum_z += numpy.sum( counter * geom.glweights )
    # Reduce the result across all MPI processes
    return MPI.COMM_WORLD.allreduce(local_sum_z)

def global_integral_3d_alt(field, geom, mtrx, metric, nbsolpts, nb_elements_horiz, nb_elements_vert):
   local_sum = 0.
   # Create the 3D weight matrix with shape (z, x, y)
   W_xyz = numpy.zeros((nbsolpts, nbsolpts, nbsolpts))

   for idx in range(nbsolpts):
      W_xyz[idx, :, :] = mtrx.quad_weights * geom.glweights[idx]

   for layer in range(nb_elements_vert):
      min_layer, max_layer = layer * nbsolpts + numpy.array([0, nbsolpts])
      for line in range(nb_elements_horiz):
         min_lin, max_lin = line * nbsolpts + numpy.array([0, nbsolpts])
         for column in range(nb_elements_horiz):
            min_col, max_col = column * nbsolpts + numpy.array([0, nbsolpts])
            local_sum += numpy.sum( field[min_layer:max_layer,min_lin:max_lin,min_col:max_col] * metric.sqrtG[min_layer:max_layer,min_lin:max_lin,min_col:max_col] * W_xyz )
   return MPI.COMM_WORLD.allreduce(local_sum)


