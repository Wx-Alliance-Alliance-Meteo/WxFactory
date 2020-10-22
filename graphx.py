import mayavi.mlab
import matplotlib.pyplot as plt
import mpi4py
import mpi4py.MPI
import numpy
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

# def plot_grid(geom):
#    '''
#       Display the grid of elements and the position of the solution points
#    '''
#    elem_x = mpi4py.MPI.COMM_WORLD.gather(geom.elem_cartX.T, root=0)
#    elem_y = mpi4py.MPI.COMM_WORLD.gather(geom.elem_cartY.T, root=0)
#    elem_z = mpi4py.MPI.COMM_WORLD.gather(geom.elem_cartZ.T, root=0)
#
#    glb_x = mpi4py.MPI.COMM_WORLD.gather(geom.cartX.T, root=0)
#    glb_y = mpi4py.MPI.COMM_WORLD.gather(geom.cartY.T, root=0)
#    glb_z = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ.T, root=0)
#
#    if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
#       fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
#
#       for f in range(nbfaces):
#          for i in range(len(elem_x[f])):
#             mayavi.mlab.plot3d(elem_x[f][i], elem_y[f][i], elem_z[f][i], tube_radius = 0.003)
#             mayavi.mlab.plot3d([a[i] for a in elem_x[f]], [a[i] for a in elem_y[f]], [a[i] for a in elem_z[f]], tube_radius = 0.003)
#
#          s = mayavi.mlab.points3d(
#             glb_x[f], glb_y[f], glb_z[f],
#             color = (1.0, 1.0, 0.0),
#             mode = "sphere",
#             scale_factor = 0.008,
#             )
#       (_,_,dist,_) = mayavi.mlab.view()
#
#       mayavi.mlab.view(azimuth=270, elevation=90, distance=dist)
#
#       mayavi.mlab.show()


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


# def plot_field_pair(geom1, field1, geom2, field2):
#
#
#    def plot_single_field(geom, field, fig_id, minmax = None):
#       glb_x     = mpi4py.MPI.COMM_WORLD.gather(geom.cartX.T, root=0)
#       glb_y     = mpi4py.MPI.COMM_WORLD.gather(geom.cartY.T, root=0)
#       glb_field = mpi4py.MPI.COMM_WORLD.gather(field.T, root=0)
#       glb_z     = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ.T, root=0)
#
#       max_val = -float("inf")
#       min_val = float("inf")
#
#       if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
#
#          if minmax is None:
#             for f in range(nbfaces):
#                face_min = glb_field[f].min()
#                face_max = glb_field[f].max()
#                if face_max > max_val:
#                   max_val = face_max
#                if face_min < min_val:
#                   min_val = face_min
#
#             min_val *= 1.03
#             max_val *= 1.03
#
#          else:
#             min_val = minmax[0]
#             max_val = minmax[1]
#
#          fig = mayavi.mlab.figure(fig_id, size=(800, 800), bgcolor=(0,0,0))
#          for f in range(nbfaces):
#             s = mayavi.mlab.mesh(glb_x[f], glb_y[f], glb_z[f], scalars=glb_field[f], colormap="jet", vmin=min_val, vmax=max_val)
#
#          mayavi.mlab.view(azimuth=270, elevation=90, distance=dist)
#          (_,_,dist,_) = mayavi.mlab.view()
#
#          mayavi.mlab.colorbar()
#
#       if minmax is None:
#           return (min_val, max_val)
#       else:
#           return minmax
#
#    min_val, max_val = plot_single_field(geom1, field1, 0)
#    plot_single_field(geom2, field2, 1, [min_val, max_val])
#
#
#    if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
#
#       mayavi.mlab.show()


# def plot_times(comm, timers):
#
#    rank = mpi4py.MPI.COMM_WORLD.Get_rank()
#
#    all_timers = [comm.gather(timers[i], root = 0) for i in range(len(timers.timers))]
#
#    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
#
#       plt.ion()
#    if rank == 0:
#       fig, ax = plt.subplots()
#
#       for i, t_list in enumerate(all_timers):
#          for j, t in enumerate(t_list):
#             starts = numpy.array(t.start_times) - timers[0].initial_time
#             stops  = starts + numpy.array(t.times)
#             y = [j/2.0 for x in range(len(starts))]
#             plt.hlines(y, starts, stops, color = colors[i], lw = 30)
#
#       #plt.show()
#
#       plt.pause(0)
