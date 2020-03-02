import mayavi.mlab
import mpi4py.MPI
import numpy

#import cartopy.crs
#import matplotlib.pyplot

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

def image_field(geom, field, filename):
   return # TODO : bug au cmc ...
#
#   lon     = numpy.array( mpi4py.MPI.COMM_WORLD.gather(geom.lon, root=0) )
#   lat     = numpy.array( mpi4py.MPI.COMM_WORLD.gather(geom.lat, root=0) )
#   data    = numpy.array( mpi4py.MPI.COMM_WORLD.gather(field, root=0) )
#
#   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
#
#      fig = matplotlib.pyplot.figure(figsize=[12.8,9.6])
#      ax = matplotlib.pyplot.axes(projection=cartopy.crs.PlateCarree())
#
      # Have to use the same range for all tiles!
      # cannot just pass None to plt.pcolormesh()
#      vmin = numpy.amin(data)
#      vmax = numpy.amax(data)
#      print((vmin,vmax)); exit(0)
#
#      vmin,vmax = (8100, 10500) # case 6
#      vmin,vmax = (0.4, 1.6) # vortex
#
#      for i in range(6):
         # 6 tiles have the same color configuration so we only return one QuadMesh object
#         im = ax.pcolormesh(numpy.degrees(lon[i]), numpy.degrees(lat[i]), data[i], vmin=vmin, vmax=vmax, transform=cartopy.crs.PlateCarree(), cmap='jet')
#
#      cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)
#      cbar.set_label("height (m)",)
#
#      ax.coastlines(alpha=0.3)
#
#      matplotlib.pyplot.show()
#      matplotlib.pyplot.savefig(filename) #, bbox_inches="tight")
#      matplotlib.pyplot.close(fig)
#
#   return

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

   glb_x = mpi4py.MPI.COMM_WORLD.gather(geom.cartX, root=0)
   glb_y = mpi4py.MPI.COMM_WORLD.gather(geom.cartY, root=0)
   glb_z = mpi4py.MPI.COMM_WORLD.gather(geom.cartZ, root=0)
   glb_lon = mpi4py.MPI.COMM_WORLD.gather(geom.lon, root=0)
   glb_lat = mpi4py.MPI.COMM_WORLD.gather(geom.lat, root=0)
   glb_u = mpi4py.MPI.COMM_WORLD.gather(u, root=0)
   glb_v = mpi4py.MPI.COMM_WORLD.gather(v, root=0)

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      fig = mayavi.mlab.figure(0, size=(800, 800), bgcolor=(0, 0, 0))
      for f in range(nbfaces):
         xdot = -glb_u[f] * numpy.sin(glb_lon[f]) - glb_v[f] * numpy.cos(glb_lon[f]) * numpy.sin(glb_lat[f])
         ydot =  glb_u[f] * numpy.cos(glb_lon[f]) - glb_v[f] * numpy.sin(glb_lon[f]) * numpy.sin(glb_lat[f])
         zdot =  glb_v[f] * numpy.cos(glb_lat[f])
         mayavi.mlab.quiver3d(glb_x[f], glb_y[f], glb_z[f], xdot, ydot, zdot)
      mayavi.mlab.show()
