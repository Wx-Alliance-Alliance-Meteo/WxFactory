import mayavi.mlab
import mpi4py
import mpi4py.MPI
import pickle

from definitions import *

def plot_sphere(geom):
   glb_x = mpi4py.MPI.COMM_WORLD.gather(geom.cartX.T, root=0)
   glb_y = mpi4py.MPI.COMM_WORLD.gather(geom.cartY.T, root=0)
   glb_z = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ.T, root=0)

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
      for f in range(nbfaces):
         mayavi.mlab.mesh(glb_x[f], glb_y[f], glb_z[f])
      mayavi.mlab.show()


def plot_field(geom, field):
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


def plot_array(array):
   rank = mpi4py.MPI.COMM_WORLD.Get_rank()

   all_arrays = mpi4py.MPI.COMM_WORLD.gather(array, root=0)


   if rank == 0:
      print(f'Doing the plotting')
      import matplotlib.pyplot as plt
      import numpy as np

      z = np.zeros_like(all_arrays[0])
      c1 = np.vstack((z, all_arrays[3], z))
      c2 = np.vstack((all_arrays[4], all_arrays[0], all_arrays[5]))
      c3 = np.vstack((z, all_arrays[1], z))
      c4 = np.vstack((z, all_arrays[2], z))
      common = np.hstack((c1, c2, c3, c4))

      # fig, ((_, p4, _, _), (p3, p0, p1, p2), (_, p5, _, _)) = plt.subplots(3, 4, figsize = (16, 9))

      # for ax, arr in zip([p0, p1, p2, p3, p4, p5], all_arrays):
      #    ax.imshow(array)

      plt.xticks(ticks = np.arange(common.shape[0]))
      plt.yticks(ticks = np.arange(common.shape[1]))
      plt.imshow(common, interpolation = 'nearest')

      plt.show()

   mpi4py.MPI.COMM_WORLD.Barrier()
