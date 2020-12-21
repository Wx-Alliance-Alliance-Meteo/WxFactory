import mpi4py
import netCDF4
import numpy
import math
import os
import time

from diagnostic import relative_vorticity, potential_vorticity
from definitions import idx_h, idx_hu1, idx_hu2, idx_u1, idx_u2
from winds import contra2wind

def output_init(geom, param):
   """ Initialise the netCDF4 file."""

   # creating the netcdf files
   global ncfile
   os.makedirs(os.path.dirname(param.output_file), exist_ok=True)
   ncfile = netCDF4.Dataset(param.output_file, 'w', format='NETCDF4', parallel = True)

   # write general attributes
   ncfile.history = 'Created ' + time.ctime(time.time())
   ncfile.description = 'GEF Model'
   ncfile.details = 'Cubed-sphere coordinates, Gauss-Legendre collocated grid'

   # create dimensions
   ni, nj= geom.lat.shape
   ncfile.createDimension('time', None) # unlimited
   ncfile.createDimension('npe', mpi4py.MPI.COMM_WORLD.Get_size())
   ncfile.createDimension('Ydim', ni)
   ncfile.createDimension('Xdim', nj)

   # create time axis
   tme = ncfile.createVariable('time', numpy.float64, ('time',))
   tme.units = 'hours since 1800-01-01'
   tme.long_name = 'time'
   tme.set_collective(True)

   # create tiles axis
   tile = ncfile.createVariable('npe', 'i4', ('npe'))
   tile.grads_dim = 'e'
   tile.standard_name = 'tile'
   tile.long_name = 'cubed-sphere tile'
   tile.axis = 'e'

   # create latitude axis
   yyy = ncfile.createVariable('Ydim', numpy.float64, ('Ydim'))
   yyy.long_name = 'Ydim'
   yyy.units = 'radians_north'

   # create longitude axis
   xxx = ncfile.createVariable('Xdim', numpy.float64, ('Xdim'))
   xxx.long_name = 'Xdim'
   xxx.units = 'radians_east'

   # create variable array
   lat = ncfile.createVariable('lats', numpy.float64, ('npe', 'Ydim', 'Xdim'))
   lat.long_name = 'latitude'
   lat.units = 'degrees_north'

   lon = ncfile.createVariable('lons', numpy.dtype('double').char, ('npe', 'Ydim', 'Xdim'))
   lon.long_name = 'longitude'
   lon.units = 'degrees_east'

   hhh = ncfile.createVariable('h', numpy.dtype('double').char, ('time', 'npe', 'Ydim', 'Xdim'))
   hhh.long_name = 'fluid height'
   hhh.units = 'm'
   hhh.coordinates = 'lons lats'
   hhh.grid_mapping = 'cubed_sphere'
   hhh.set_collective(True)

   if param.case_number >= 2:
      uuu = ncfile.createVariable('U', numpy.dtype('double').char, ('time', 'npe', 'Ydim', 'Xdim'))
      uuu.long_name = 'eastward_wind'
      uuu.units = 'm s-1'
      uuu.standard_name = 'eastward_wind'
      uuu.coordinates = 'lons lats'
      uuu.grid_mapping = 'cubed_sphere'
      uuu.set_collective(True)

      vvv = ncfile.createVariable('V', numpy.dtype('double').char, ('time', 'npe', 'Ydim', 'Xdim'))
      vvv.long_name = 'northward_wind'
      vvv.units = 'm s-1'
      vvv.standard_name = 'northward_wind'
      vvv.coordinates = 'lons lats'
      vvv.grid_mapping = 'cubed_sphere'
      vvv.set_collective(True)

      drv = ncfile.createVariable('RV', numpy.dtype('double').char, ('time', 'npe', 'Ydim', 'Xdim'))
      drv.long_name = 'Relative vorticity'
      drv.units = '1/(m s)'
      drv.standard_name = 'Relative vorticity'
      drv.coordinates = 'lons lats'
      drv.grid_mapping = 'cubed_sphere'
      drv.set_collective(True)

      dpv = ncfile.createVariable('PV', numpy.dtype('double').char, ('time', 'npe', 'Ydim', 'Xdim'))
      dpv.long_name = 'Potential vorticity'
      dpv.units = '1/(m s)'
      dpv.standard_name = 'Potential vorticity'
      dpv.coordinates = 'lons lats'
      dpv.grid_mapping = 'cubed_sphere'
      dpv.set_collective(True)

   rank = mpi4py.MPI.COMM_WORLD.Get_rank()

   if rank == 0:
      xxx[:] = geom.x1[:]
      yyy[:] = geom.x2[:]

   tile[rank] = rank
   lon[rank,:,:] = geom.lon * 180/math.pi
   lat[rank,:,:] = geom.lat * 180/math.pi


def output_netcdf(Q, geom, metric, mtrx, topo, step, param):
   """ Writes u,v,eta fields on every nth time step """
   rank = mpi4py.MPI.COMM_WORLD.Get_rank()

   ncfile['time'][step] = step * param.dt

   # Unpack physical variables
   h = Q[idx_h, :, :] + topo.hsurf
   ncfile['h'][step, rank, :, :] = h

   if param.case_number >= 2: # Shallow water
      u1 = Q[idx_hu1,:,:] / h
      u2 = Q[idx_hu2,:,:] / h
      u, v = contra2wind(u1, u2, geom)
      rv = relative_vorticity(u1, u2, geom, metric, mtrx, param)
      pv = potential_vorticity(h, u1, u2, geom, metric, mtrx, param)

      ncfile['U'][step, rank, :, :] = u
      ncfile['V'][step, rank, :, :] = v
      ncfile['RV'][step, rank, :, :] = rv
      ncfile['PV'][step, rank, :, :] = pv

def output_finalize():
   """ Finalise the output netCDF4 file."""
   if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
      ncfile.close()
