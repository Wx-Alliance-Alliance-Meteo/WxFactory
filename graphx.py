import mayavi.mlab

from constants import *

def plot_sphere(geom):
   mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
   for f in range(nbfaces):
      mayavi.mlab.mesh(geom.cartX.T[f], geom.cartY.T[f], geom.cartZ.T[f])
   mayavi.mlab.show()

   # TODO : Dans une classe graphic ???
def plot_field(geom, field):
#   plt.plot_source = []

   min_val = float("inf")
   max_val = -float("inf")
   for f in range(nbfaces):
      face_min = field[f].min()
      face_max = field[f].max()
      if face_max > max_val:
         max_val = face_max
      if face_min < min_val:
         min_val = face_min

   fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0,0,0))
   for f in range(nbfaces):
      s = mayavi.mlab.mesh(geom.cartX.T[f], geom.cartY.T[f], geom.cartZ.T[f], scalars=field.T[f], colormap="jet", vmin=min_val, vmax=max_val)
#      plt.plot_source.append(s)

   (_,_,dist,_) = mayavi.mlab.view()
   mayavi.mlab.view(azimuth=270, elevation=90, distance=dist)

   mayavi.mlab.colorbar()
   mayavi.mlab.show()

#https://stackoverflow.com/questions/39840638/update-mayavi-plot-in-loop
#def plot_update(geom, field):
#   for f in range(nbfaces):
#      plt.plot_source[f].mlab_source.scalars = field[f]

#def plot_vect(geom, u, v): # TODO : cov, contra
#
#   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
#      fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0, 0, 0))
#      for f in range(nbfaces):
#         xdot = -glb_u[f] * numpy.sin(glb_lon[f]) - glb_v[f] * numpy.cos(glb_lon[f]) * numpy.sin(glb_lat[f])
#         ydot =  glb_u[f] * numpy.cos(glb_lon[f]) - glb_v[f] * numpy.sin(glb_lon[f]) * numpy.sin(glb_lat[f])
#         zdot =  glb_v[f] * numpy.cos(glb_lat[f])
#         mayavi.mlab.quiver3d(glb_x[f], glb_y[f], glb_z[f], xdot, ydot, zdot);
#      mayavi.mlab.show()
