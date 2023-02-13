import mpi4py
import mpi4py.MPI
import numpy
import pickle
import matplotlib.pyplot

try:
   import mayavi.mlab
except ModuleNotFoundError:
   if mpi4py.MPI.COMM_WORLD.Get_size() > 1:
      print(f'WARNING: Could not import mayavi module. There will be a crash if you try to plot stuff on the cubed sphere')

from common.definitions import nbfaces

plot_index = 0

def plot_sphere(geom):
   glb_x = mpi4py.MPI.COMM_WORLD.gather(geom.cartX.T, root=0)
   glb_y = mpi4py.MPI.COMM_WORLD.gather(geom.cartY.T, root=0)
   glb_z = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ.T, root=0)

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
      for f in range(nbfaces):
         mayavi.mlab.mesh(glb_x[f], glb_y[f], glb_z[f])
      mayavi.mlab.show()

def plot_level(geom, field, lvl):
   glb_x     = mpi4py.MPI.COMM_WORLD.gather(geom.cartX[lvl].T, root=0)
   glb_y     = mpi4py.MPI.COMM_WORLD.gather(geom.cartY[lvl].T, root=0)
   glb_z     = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ[lvl].T, root=0)
   glb_field = mpi4py.MPI.COMM_WORLD.gather(field[lvl].T, root=0)

   ptopo_size = mpi4py.MPI.COMM_WORLD.Get_size()
   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      min_val = float("inf")
      max_val = -float("inf")
      for f in range(ptopo_size):
         face_min = glb_field[f].min()
         face_max = glb_field[f].max()
         if face_max > max_val:
            max_val = face_max
         if face_min < min_val:
            min_val = face_min

      fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
      for f in range(ptopo_size):
         s = mayavi.mlab.mesh(glb_x[f], glb_y[f], glb_z[f], scalars=glb_field[f], colormap="jet", vmin=min_val, vmax=max_val)

      (_,_,dist,_) = mayavi.mlab.view()
      mayavi.mlab.view(azimuth=270, elevation=90, distance=dist)

      mayavi.mlab.colorbar()
      mayavi.mlab.show()

   mpi4py.MPI.COMM_WORLD.Barrier()



def plot_field(geom, field, filename=None):
   glb_x     = mpi4py.MPI.COMM_WORLD.gather(geom.cartX.T, root=0)
   glb_y     = mpi4py.MPI.COMM_WORLD.gather(geom.cartY.T, root=0)
   glb_z     = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ.T, root=0)
   glb_field = mpi4py.MPI.COMM_WORLD.gather(field.T, root=0)

   ptopo_size = mpi4py.MPI.COMM_WORLD.Get_size()
   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      min_val = float("inf")
      max_val = -float("inf")
      for f in range(ptopo_size):
         face_min = glb_field[f].min()
         face_max = glb_field[f].max()
         if face_max > max_val:
            max_val = face_max
         if face_min < min_val:
            min_val = face_min

      fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
      for f in range(ptopo_size):
         s = mayavi.mlab.mesh(glb_x[f], glb_y[f], glb_z[f], scalars=glb_field[f], colormap="jet", vmin=min_val, vmax=max_val)

      (_,_,dist,_) = mayavi.mlab.view()
      mayavi.mlab.view(azimuth=270, elevation=90, distance=dist)

      mayavi.mlab.colorbar()

      if filename is not None:
         mayavi.mlab.savefig(filename)
      else:
         mayavi.mlab.show()

   mpi4py.MPI.COMM_WORLD.Barrier()


def plot_vector_field(geom, field_x1, field_x2, filename=None):
   glb_x        = mpi4py.MPI.COMM_WORLD.gather(geom.cartX.T, root=0)
   glb_y        = mpi4py.MPI.COMM_WORLD.gather(geom.cartY.T, root=0)
   glb_z        = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ.T, root=0)
   glb_field_x1 = mpi4py.MPI.COMM_WORLD.gather(field_x1.T, root=0)
   glb_field_x2 = mpi4py.MPI.COMM_WORLD.gather(field_x2.T, root=0)

   ptopo_size = mpi4py.MPI.COMM_WORLD.Get_size()
   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      min_val = float("inf")
      max_val = -float("inf")
      for f in range(ptopo_size):
         face_min = min(glb_field_x1[f].min(), glb_field_x2[f].min())
         face_max = max(glb_field_x1[f].max(), glb_field_x2[f].max())
         if face_max > max_val:
            max_val = face_max
         if face_min < min_val:
            min_val = face_min

      fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
      for f in range(ptopo_size):
         glb_field = glb_field_x1[f].copy()
         n_rows = glb_field.shape[0]
         for i in range(n_rows):
            if i < n_rows / 2:
               glb_field[i][i:n_rows-i] = glb_field_x2[f][i][i:n_rows-i]
            else:
               glb_field[i][n_rows-i:i] = glb_field_x2[f][i][n_rows-i:i]

         s = mayavi.mlab.mesh(glb_x[f], glb_y[f], glb_z[f], scalars=glb_field, colormap="jet", vmin=min_val, vmax=max_val)

      (_,_,dist,_) = mayavi.mlab.view()
      mayavi.mlab.view(azimuth=270, elevation=90, distance=dist)

      mayavi.mlab.colorbar()

      if filename is not None:
         mayavi.mlab.savefig(filename)
      else:
         mayavi.mlab.show()

   mpi4py.MPI.COMM_WORLD.Barrier()


def plot_field_from_file(geom_prefix, field_prefix):
   rank = mpi4py.MPI.COMM_WORLD.Get_rank()
   suffix = '{:04d}.dat'.format(rank)
   geom_filename = geom_prefix + suffix
   field_filename = field_prefix + suffix

   geom  = pickle.load(open(geom_filename, 'rb'))
   field = pickle.load(open(field_filename, 'rb'))

   # if rank == 0:
   #    print('field = {}'.format(field[0,:,:]))

   plot_field(geom, field[0,:,:]**2)


def plot_array(array, filename=None):
   rank = mpi4py.MPI.COMM_WORLD.Get_rank()

   all_arrays = mpi4py.MPI.COMM_WORLD.gather(array, root=0)

   if rank == 0:
      print(f'Doing the plotting')
      import matplotlib.pyplot as plt
      import numpy as np

      z = np.zeros_like(all_arrays[0])
      c1 = np.vstack((z, np.flipud(all_arrays[3]), z))
      c2 = np.vstack((np.flipud(all_arrays[4]), np.flipud(all_arrays[0]), np.flipud(all_arrays[5])))
      c3 = np.vstack((z, np.flipud(all_arrays[1]), z))
      c4 = np.vstack((z, np.flipud(all_arrays[2]), z))
      common = np.hstack((c1, c2, c3, c4))

      plt.clf()
      ax = plt.gca()

      im = ax.imshow(common, interpolation='nearest')
      cbar = ax.figure.colorbar(im, ax=ax)
      ax.set_xticks(ticks=np.arange(0, common.shape[0], 2))
      ax.set_yticks(ticks=np.arange(0, common.shape[1], 2))

      plt.tight_layout()

      if filename is None:
         plt.show()
      else:
         plt.savefig(filename)

   mpi4py.MPI.COMM_WORLD.Barrier()

def image_field(geom, field, filename, vmin, vmax, n):
   fig, ax = matplotlib.pyplot.subplots()
      
   cmap = matplotlib.pyplot.contourf(geom.X, geom.Z, field, cmap='jet', levels=numpy.linspace(vmin,vmax,n), extend="both")
   ax.set_aspect('equal', 'box')

   cbar = fig.colorbar(cmap, ax=ax, orientation='vertical', shrink=0.5)
   cbar.set_label("K",)

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

      cmap = ax.contourf(geom.X, geom.Z, vals, levels=levels)
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