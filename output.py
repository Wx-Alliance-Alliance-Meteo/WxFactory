import mpi4py
import netCDF4
import numpy 
import math
import time

def output_nc_ini(geom):
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
      ncfile.createDimension('face', 6)
      ncfile.createDimension('X', ni)
      ncfile.createDimension('Y', nj)
   
      # create time axis
      tme = ncfile.createVariable('time', 'i8', ('time',))
      tme.long_name = 'time'
      tme.units = 'hours since 1990-01-01 00:00:00'
      tme.calendar = 'standard'
      tme.axis = 'T'

      # create faces axis
      face = ncfile.createVariable('face', 'i4', ('face'))
      face.standard_name = 'face'
      face.long_name = 'Cubed-sphere face'
      face.axis = 'face'

      # create latitude axis
      yyy = ncfile.createVariable('Y', numpy.dtype('double').char, ('Y'))
      yyy.standard_name = 'Y'
      yyy.long_name = 'Y'
      yyy.units = 'radians_north'
      yyy.axis = 'Y'

      # create longitude axis
      xxx = ncfile.createVariable('X', numpy.dtype('double').char, ('X'))
      xxx.standard_name = 'X'
      xxx.long_name = 'X'
      xxx.units = 'radians_east'
      xxx.axis = 'X'

      # create variable array
      lat = ncfile.createVariable('lat', numpy.dtype('double').char, ('face', 'X', 'Y'))
      lat.standard_name = 'latitude'
      lat.long_name = 'latitude'
      lat.units = 'degrees_north'

      lon = ncfile.createVariable('lon', numpy.dtype('double').char, ('face', 'X', 'Y'))
      lon.standard_name = 'longitude'
      lon.long_name = 'longitude'
      lon.units = 'degrees_east'

      eta = ncfile.createVariable('eta', numpy.dtype('double').char, ('time', 'face', 'X', 'Y'))
      eta.standard_name = 'h'
      eta.long_name = 'fluid height'
      eta.units = 'm'

      xxx[:] = geom.x1[:]
      yyy[:] = geom.x2[:]
      
      lon[:,:,:] = glb_lon[:,:,:] * 180/math.pi
      lat[:,:,:] = glb_lat[:,:,:] * 180/math.pi
   

def output_nc(u,v,eta,t, step):
   """ Writes u,v,eta fields on every nth time step """
   
   glb_h = numpy.array( mpi4py.MPI.COMM_WORLD.gather(eta, root=0) )

    #TODO issue, use unlimited time dimension or not?
#    ncfile['u'][step,:,:] = u
#    ncfile['v'][step,:,:] = v

   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:

      ncfile['time'][step] = t
      ncfile['eta'][step,:,:,:] = glb_h


def output_nc_fin():
   """ Finalise the output netCDF4 file."""
   ncfile.close()
