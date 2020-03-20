import mpi4py
import netCDF4
import numpy 
import math
import time

from definitions import idx_h, idx_hu1, idx_hu2, idx_u1, idx_u2
from winds import contra2wind

def output_init(geom):
   """ Initialise the netCDF4 file."""

   glb_lon = numpy.array( mpi4py.MPI.COMM_WORLD.gather(geom.lon, root=0) )
   glb_lat = numpy.array( mpi4py.MPI.COMM_WORLD.gather(geom.lat, root=0) )

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:

      # creating the netcdf files
      global ncfile
      ncfile = netCDF4.Dataset('/tmp/out.nc', 'w', format='NETCDF4')
   
      # write general attributes
      ncfile.history = 'Created ' + time.ctime(time.time())
      ncfile.description = 'GEF Model'
      ncfile.details = 'Cubed-sphere coordinates, Gauss-Legendre collocated grid'
   
      # create dimensions
      ni, nj= geom.lat.shape
      ncfile.createDimension('time', None) # unlimited
      ncfile.createDimension('nf', 6)
      ncfile.createDimension('Ydim', nj)
      ncfile.createDimension('Xdim', ni)
   
      # create time axis
      tme = ncfile.createVariable('time', numpy.float64, ('time',))
      tme.units = 'hours since 1800-01-01'
      tme.long_name = 'time'

      # create faces axis
      face = ncfile.createVariable('nf', 'i4', ('nf'))
      face.grads_dim = 'e'
      face.standard_name = 'face'
      face.long_name = 'cubed-sphere face'
      face.axis = 'e'

      # create latitude axis
      yyy = ncfile.createVariable('Ydim', numpy.float64, ('Ydim'))
      yyy.long_name = 'Ydim'
      yyy.units = 'radians_north'

      # create longitude axis
      xxx = ncfile.createVariable('Xdim', numpy.float64, ('Xdim'))
      xxx.long_name = 'Xdim'
      xxx.units = 'radians_east'

      # create variable array
      lat = ncfile.createVariable('lats', numpy.float64, ('nf', 'Xdim', 'Ydim'))
      lat.long_name = 'latitude'
      lat.units = 'degrees_north'

      lon = ncfile.createVariable('lons', numpy.dtype('double').char, ('nf', 'Xdim', 'Ydim'))
      lon.long_name = 'longitude'
      lon.units = 'degrees_east'

      hhh = ncfile.createVariable('h', numpy.dtype('double').char, ('time', 'nf', 'Xdim', 'Ydim'))
      hhh.long_name = 'fluid height'
      hhh.units = 'm'
      hhh.coordinates = 'lons lats'
      hhh.grid_mapping = 'cubed_sphere'
      
      uuu = ncfile.createVariable('U', numpy.dtype('double').char, ('time', 'nf', 'Xdim', 'Ydim'))
      uuu.long_name = 'eastward_wind'
      uuu.units = 'm s-1'
      uuu.standard_name = 'eastward_wind'
      uuu.coordinates = 'lons lats'
      uuu.grid_mapping = 'cubed_sphere'

      vvv = ncfile.createVariable('V', numpy.dtype('double').char, ('time', 'nf', 'Xdim', 'Ydim'))
      vvv.long_name = 'northward_wind'
      vvv.units = 'm s-1'
      vvv.standard_name = 'northward_wind'
      vvv.coordinates = 'lons lats'
      vvv.grid_mapping = 'cubed_sphere'

      xxx[:] = geom.x1[:]
      yyy[:] = geom.x2[:]
      
      lon[:,:,:] = glb_lon[:,:,:] * 180/math.pi
      lat[:,:,:] = glb_lat[:,:,:] * 180/math.pi
      
   

def output_netcdf(Q, geom, topo, step, param):
   """ Writes u,v,eta fields on every nth time step """
   # Unpack physical variables
   h = Q[idx_h, :, :] + topo.hsurf

   advection_only = ( param.case_number <= 1 )

   if advection_only:
      u1 = Q[idx_u1, :, :]
      u2 = Q[idx_u2, :, :]
   else:
      u1 = Q[idx_hu1,:,:] / h
      u2 = Q[idx_hu2,:,:] / h

   u, v = contra2wind(u1, u2, geom)
#   u, v = u1, u2

   # Assemble global array
   glb_h = numpy.array( mpi4py.MPI.COMM_WORLD.gather(h, root=0) )
   glb_u = numpy.array( mpi4py.MPI.COMM_WORLD.gather(u, root=0) )
   glb_v = numpy.array( mpi4py.MPI.COMM_WORLD.gather(v, root=0) )

   
   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:

      ncfile['time'][step] = step * param.dt
      ncfile['h'][step,:,:,:] = glb_h
      ncfile['U'][step,:,:,:] = glb_u
      ncfile['V'][step,:,:,:] = glb_v


def output_finalize():
   """ Finalise the output netCDF4 file."""
   ncfile.close()
