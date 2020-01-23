import mayavi.mlab
import mpi4py.MPI
import numpy

from constants import *

def plot_sphere(geom):
   glb_x = mpi4py.MPI.COMM_WORLD.gather(geom.cartX.T, root=0)
   glb_y = mpi4py.MPI.COMM_WORLD.gather(geom.cartY.T, root=0)
   glb_z = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ.T, root=0)

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
      for f in range(nbfaces):
         mayavi.mlab.mesh(glb_x[f], glb_y[f], glb_z[f])
      mayavi.mlab.show()

   # TODO : Dans une classe graphic ???
def plot_field(geom, field):
   glb_x     = mpi4py.MPI.COMM_WORLD.gather(geom.cartX.T, root=0)
   glb_y     = mpi4py.MPI.COMM_WORLD.gather(geom.cartY.T, root=0)
   glb_z     = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ.T, root=0)
   glb_field = mpi4py.MPI.COMM_WORLD.gather(field.T, root=0)

#   plt.plot_source = []

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      min_val = float("inf")
      max_val = -float("inf")
      for f in range(nbfaces):
         face_min = glb_field[f].min()
         face_max = glb_field[f].max()
         if face_max > max_val:
            max_val = face_max
         if face_min < min_val:
            min_val = face_min

      fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
      for f in range(nbfaces):
         s = mayavi.mlab.mesh(glb_x[f], glb_y[f], glb_z[f], scalars=glb_field[f], colormap="jet", vmin=min_val, vmax=max_val)
   #      plt.plot_source.append(s)

      (_,_,dist,_) = mayavi.mlab.view()
      mayavi.mlab.view(azimuth=270, elevation=90, distance=dist)

      mayavi.mlab.colorbar()
      mayavi.mlab.show()

#https://stackoverflow.com/questions/39840638/update-mayavi-plot-in-loop
#def plot_update(geom, field):
#   for f in range(nbfaces):
#      plt.plot_source[f].mlab_source.scalars = field[f]

def plot_uv(geom, u, v): # TODO : cov, contra
   fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0, 0, 0))
   for f in range(nbfaces):
      xdot = -u[:,:,f] * numpy.sin(geom.lon[:,:,f]) - v[:,:,f] * numpy.cos(geom.lon[:,:,f]) * numpy.sin(geom.lat[:,:,f])
      ydot =  u[:,:,f] * numpy.cos(geom.lon[:,:,f]) - v[:,:,f] * numpy.sin(geom.lon[:,:,f]) * numpy.sin(geom.lat[:,:,f])
      zdot =  v[:,:,f] * numpy.cos(geom.lat[:,:,f])
      mayavi.mlab.quiver3d(geom.cartX.T[f], geom.cartY.T[f], geom.cartZ.T[f], xdot.T, ydot.T, zdot.T)
   mayavi.mlab.show()
