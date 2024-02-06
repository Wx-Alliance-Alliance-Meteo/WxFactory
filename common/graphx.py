from mpi4py import MPI
import numpy
import pickle
import matplotlib.pyplot

plot_index = 0

def plot_field_from_file(geom_prefix, field_prefix):
   rank = MPI.COMM_WORLD.Get_rank()
   suffix = '{:04d}.dat'.format(rank)
   geom_filename = geom_prefix + suffix
   field_filename = field_prefix + suffix

   geom  = pickle.load(open(geom_filename, 'rb'))
   field = pickle.load(open(field_filename, 'rb'))

   # if rank == 0:
   #    print('field = {}'.format(field[0,:,:]))

   plot_field(geom, field[0,:,:]**2)

def plot_array(array, filename=None):
   rank = MPI.COMM_WORLD.Get_rank()

   all_arrays = MPI.COMM_WORLD.gather(array, root=0)

   if rank == 0:
      print(f'Doing the plotting')

      z = numpy.zeros_like(all_arrays[0])
      c1 = numpy.vstack((z, numpy.flipud(all_arrays[3]), z))
      c2 = numpy.vstack((numpy.flipud(all_arrays[4]), numpy.flipud(all_arrays[0]), numpy.flipud(all_arrays[5])))
      c3 = numpy.vstack((z, numpy.flipud(all_arrays[1]), z))
      c4 = numpy.vstack((z, numpy.flipud(all_arrays[2]), z))
      common = numpy.hstack((c1, c2, c3, c4))

      matplotlib.pyplot.clf()
      ax = matplotlib.pyplot.gca()

      im = ax.imshow(common, interpolation='nearest')
      cbar = ax.figure.colorbar(im, ax=ax)
      ax.set_xticks(ticks=numpy.arange(0, common.shape[0], 2))
      ax.set_yticks(ticks=numpy.arange(0, common.shape[1], 2))

      matplotlib.pyplot.tight_layout()

      if filename is None:
         matplotlib.pyplot.show()
      else:
         matplotlib.pyplot.savefig(filename)

   MPI.COMM_WORLD.Barrier()

def image_field(geom: 'Cartesian2D', field: numpy.ndarray, filename: str, vmin: float, vmax: float, n: int, \
                label: str = 'K', colormap: str = 'jet'):
   fig, ax = matplotlib.pyplot.subplots()

   if not geom.xperiodic:
      cmap = matplotlib.pyplot.contourf(geom.X1, geom.X3, field, cmap=colormap,
                                       levels=numpy.linspace(vmin,vmax,n), extend="both")
   else:
      # X1 = numpy.append(geom.X1[:, -1:], geom.X1, axis=1)
      # print(f'geom x1: \n{geom.X1[:, :2]}')
      # print(f'x1: {X1[:, :3]}')
      # raise ValueError
      X1 = numpy.append(numpy.append(geom.X1[:, -2:], geom.X1, axis=1), geom.X1[:, :2], axis=1)
      X1[:,  1] = 2*X1[:,  2] - X1[:,  3]
      X1[:,  0] = 2*X1[:,  1] - X1[:,  2]
      X1[:, -2] = 2*X1[:, -3] - X1[:, -4]
      X1[:, -1] = 2*X1[:, -2] - X1[:, -3]
      X3 = numpy.append(numpy.append(geom.X3[:, -2:], geom.X3, axis=1), geom.X3[:, :2], axis=1)
      X3[:,  1] = 2*X3[:,  2] - X3[:,  3]
      X3[:,  0] = 2*X3[:,  1] - X3[:,  2]
      X3[:, -2] = 2*X3[:, -3] - X3[:, -4]
      X3[:, -1] = 2*X3[:, -2] - X3[:, -3]
      f  = numpy.append(numpy.append(field[:, -2:], field, axis=1), field[:, :2], axis=1)
      cmap = matplotlib.pyplot.contourf(X1, X3, f, cmap=colormap,
                                       levels=numpy.linspace(vmin,vmax,n), extend="both")
   ax.set_aspect('auto', 'box')

   cbar = fig.colorbar(cmap, ax=ax, orientation='vertical', shrink=0.5)
   cbar.set_label(label, )

   matplotlib.pyplot.savefig(filename)
   matplotlib.pyplot.close(fig) 

   return

def print_mountain(x1, x3, mountain: numpy.ndarray,
                   normals_x: numpy.ndarray = None, normals_z: numpy.ndarray = None,
                   filename: str = None):
   # nb_elem = geom.nb_elements_relief_layer * geom.nbsolpts

   if not ((normals_x is None) == (normals_z is None)):
      raise ValueError(f'Either provide both normal arrays or none of them')

   fig, ax_mtn = matplotlib.pyplot.subplots()
   # cmap = matplotlib.pyplot.contourf(geom.X1[:nb_elem, :], geom.X3[:nb_elem, :], mountain[:nb_elem, :], levels=2)
   # matplotlib.pyplot.imshow(numpy.flip(mountain[:nb_elem, :], axis=0), interpolation='nearest')
   ax_mtn.scatter(x1, x3, c=mountain, marker='s')

   # ax_mtn.arrow(10, 10, 10, 10)
   length_factor = 30.0
   if normals_x is not None:
      for i in range(mountain.shape[0]):
         for j in range(mountain.shape[1]):
            if (mountain[i, j] > 0) and (normals_x[i, j] ** 2 + normals_z[i, j] ** 2 > 1e-1):
                  ax_mtn.arrow(x1[i, j], x3[i, j],
                               normals_x[i, j] * length_factor, normals_z[i, j] * length_factor,
                               head_width=40.0, head_length=10.0)

   if filename is None:
      filename = 'mountain.png'

   matplotlib.pyplot.savefig(filename)
   matplotlib.pyplot.close(fig) 

   return

def print_residual_per_variable(geom, field, filename = None):

   num_levels = 20

   fig, axes = matplotlib.pyplot.subplots(2, 2, sharex=True, sharey=True)

   def plot_var(ax, vals, title):
      # Always include 0 (?), avoid null range 
      # minval = min(vals.min() - 1e-15, 0.0)
      # maxval = max(vals.max() + 1e-15, 0.0)
      minval = vals.min() - 1e-15
      maxval = vals.max() + 1e-15
      if minval < 0.0 and maxval > 0.0:
         ratio = maxval  / (maxval - minval)
         num_pos = int(numpy.rint((num_levels) * ratio))
         num_neg = num_levels - num_pos
         minval = -maxval / num_pos * num_neg

      levels = numpy.linspace(minval, maxval, num_levels)

      cmap = ax.contourf(geom.X1, geom.X3, vals, levels=levels)
      ax.set_title(title)
      cbar = fig.colorbar(cmap, ax=ax, orientation='vertical', format = '%8.1e')
      cbar.set_label('Residual',)

   plot_var(axes[0][0], field[RHO], 'Rho')
   plot_var(axes[0][1], field[RHO_THETA], 'Rho-theta')
   plot_var(axes[1][0], field[RHO_U], 'Rho-u')
   plot_var(axes[1][1], field[RHO_W], 'Rho-w')

   # cbar = fig.colorbar(cmap, ax=axes, orientation='vertical')
   # cbar.set_label('Residual',)

   global plot_index

   fn = filename
   if fn is None: fn = f'res_plot/residual{plot_index:04d}.png'
   matplotlib.pyplot.savefig(fn)
   matplotlib.pyplot.close(fig)

   plot_index += 1

   return
