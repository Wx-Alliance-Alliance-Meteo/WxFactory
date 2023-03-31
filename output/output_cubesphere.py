from mpi4py import MPI
import netCDF4
import numpy
import math
import time

from common.definitions import *
from geometry           import contra2wind_2d, contra2wind_3d
from output.diagnostic  import relative_vorticity, potential_vorticity

def output_init(geom, param):
   """ Initialise the netCDF4 file."""

   # creating the netcdf files
   global ncfile
   ncfile = netCDF4.Dataset(param.output_file, 'w', format='NETCDF4', parallel = True)

   # write general attributes
   ncfile.history = 'Created ' + time.ctime(time.time())
   ncfile.description = 'GEF Model'
   ncfile.details = 'Cubed-sphere coordinates, Gauss-Legendre collocated grid'

   # create dimensions
   if param.equations == "shallow_water":
      ni, nj = geom.lat.shape
      grid_data = ('npe', 'Xdim', 'Ydim')
   elif param.equations == "euler":
      nk, nj, ni = geom.nk, geom.nj, geom.ni
      grid_data = ('npe', 'Zdim', 'Xdim', 'Ydim')
   else:
      print(f"Unsupported equation type {param.equations}")
      exit(1)

   grid_data2D = ('npe', 'Xdim', 'Ydim')

   ncfile.createDimension('time', None) # unlimited
   npe = MPI.COMM_WORLD.Get_size()
   ncfile.createDimension('npe', npe)
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
   yyy.axis = 'Y'
   yyy.units = 'radians_north'

   # create longitude axis
   xxx = ncfile.createVariable('Xdim', numpy.float64, ('Xdim'))
   xxx.long_name = 'Xdim'
   xxx.axis = 'X'
   xxx.units = 'radians_east'

   if param.equations == "euler":
      ncfile.createDimension('Zdim', nk)
      zzz = ncfile.createVariable('Zdim', numpy.float64, ('Zdim'))
      zzz.long_name = 'Zdim'
      zzz.axis = 'Z'
      zzz.units = 'm'

   # create variable array
   lat = ncfile.createVariable('lats', numpy.float64, grid_data2D)
   lat.long_name = 'latitude'
   lat.units = 'degrees_north'

   lon = ncfile.createVariable('lons', numpy.dtype('double').char, grid_data2D)
   lon.long_name = 'longitude'
   lon.units = 'degrees_east'

   if param.equations == "shallow_water":

      hhh = ncfile.createVariable('h', numpy.dtype('double').char, ('time', ) + grid_data)
      hhh.long_name = 'fluid height'
      hhh.units = 'm'
      hhh.coordinates = 'lons lats'
      hhh.grid_mapping = 'cubed_sphere'
      hhh.set_collective(True)

      if param.case_number >= 2:
         uuu = ncfile.createVariable('U', numpy.dtype('double').char, ('time', ) + grid_data)
         uuu.long_name = 'eastward_wind'
         uuu.units = 'm s-1'
         uuu.standard_name = 'eastward_wind'
         uuu.coordinates = 'lons lats'
         uuu.grid_mapping = 'cubed_sphere'
         uuu.set_collective(True)

         vvv = ncfile.createVariable('V', numpy.dtype('double').char, ('time', ) + grid_data)
         vvv.long_name = 'northward_wind'
         vvv.units = 'm s-1'
         vvv.standard_name = 'northward_wind'
         vvv.coordinates = 'lons lats'
         vvv.grid_mapping = 'cubed_sphere'
         vvv.set_collective(True)

         drv = ncfile.createVariable('RV', numpy.dtype('double').char, ('time', ) + grid_data)
         drv.long_name = 'Relative vorticity'
         drv.units = '1/(m s)'
         drv.standard_name = 'Relative vorticity'
         drv.coordinates = 'lons lats'
         drv.grid_mapping = 'cubed_sphere'
         drv.set_collective(True)

         dpv = ncfile.createVariable('PV', numpy.dtype('double').char, ('time', ) + grid_data)
         dpv.long_name = 'Potential vorticity'
         dpv.units = '1/(m s)'
         dpv.standard_name = 'Potential vorticity'
         dpv.coordinates = 'lons lats'
         dpv.grid_mapping = 'cubed_sphere'
         dpv.set_collective(True)

   elif param.equations == "euler":
      elev = ncfile.createVariable('elev', numpy.dtype('double').char, grid_data)
      elev.long_name = 'Elevation'
      elev.units = 'm'
      elev.standard_name = 'Elevation'
      elev.coordinates = 'lons lats'
      elev.grid_mapping = 'cubed_sphere'
      elev.set_collective(True)

      topo = ncfile.createVariable('topo', numpy.dtype('double').char, grid_data2D)
      topo.long_name = 'Topopgraphy'
      topo.units = 'm'
      topo.standard_name = 'Topography'
      topo.coordinates = 'lons lats'
      topo.grid_mapping = 'cubed_sphere'
      topo.set_collective(True)

      uuu = ncfile.createVariable('U', numpy.dtype('double').char, ('time', ) + grid_data)
      uuu.long_name = 'eastward_wind'
      uuu.units = 'm s-1'
      uuu.standard_name = 'eastward_wind'
      uuu.coordinates = 'lons lats'
      uuu.grid_mapping = 'cubed_sphere'
      uuu.set_collective(True)

      vvv = ncfile.createVariable('V', numpy.dtype('double').char, ('time', ) + grid_data)
      vvv.long_name = 'northward_wind'
      vvv.units = 'm s-1'
      vvv.standard_name = 'northward_wind'
      vvv.coordinates = 'lons lats'
      vvv.grid_mapping = 'cubed_sphere'
      vvv.set_collective(True)

      www = ncfile.createVariable('W', numpy.dtype('double').char, ('time', ) + grid_data)
      www.long_name = 'upward_air_velocity'
      www.units = 'm s-1'
      www.standard_name = 'upward_air_velocity'
      www.coordinates = 'lons lats'
      www.grid_mapping = 'cubed_sphere'
      www.set_collective(True)

      density = ncfile.createVariable('rho', numpy.dtype('double').char, ('time',) + grid_data)
      density.long_name = 'air_density'
      density.units = 'kg m-3'
      density.standard_name = 'air_density'
      density.coordinates = 'lons lats'
      density.grid_mapping = 'cubed_sphere'
      density.set_collective(True)

      potential_temp = ncfile.createVariable('theta', numpy.dtype('double').char, ('time',) + grid_data)
      potential_temp.long_name = 'air_potential_temperature'
      potential_temp.units = 'K'
      potential_temp.standard_name = 'air_potential_temperature'
      potential_temp.coordinates = 'lons lats'
      potential_temp.grid_mapping = 'cubed_sphere'
      potential_temp.set_collective(True)

      press = ncfile.createVariable('P', numpy.dtype('double').char, ('time',) + grid_data)
      press.long_name = 'air_pressure'
      press.units = 'Pa'
      press.standard_name = 'air_pressure'
      press.coordinates = 'lons lats'
      press.grid_mapping = 'cubed_sphere'
      press.set_collective(True)

      if param.case_number == 11 or param.case_number == 12:
         q1 = ncfile.createVariable('q1', numpy.dtype('double').char, ('time',) + grid_data)
         q1.long_name = 'q1'
         q1.units = 'kg m-3'
         q1.standard_name = 'Tracer q1'
         q1.coordinates = 'lons lats'
         q1.grid_mapping = 'cubed_sphere'
         q1.set_collective(True)

      if param.case_number == 11:
         q2 = ncfile.createVariable('q2', numpy.dtype('double').char, ('time',) + grid_data)
         q2.long_name = 'q2'
         q2.units = 'kg m-3'
         q2.standard_name = 'Tracer q2'
         q2.coordinates = 'lons lats'
         q2.grid_mapping = 'cubed_sphere'
         q2.set_collective(True)

         q3 = ncfile.createVariable('q3', numpy.dtype('double').char, ('time',) + grid_data)
         q3.long_name = 'q3'
         q3.units = 'kg m-3'
         q3.standard_name = 'Tracer q3'
         q3.coordinates = 'lons lats'
         q3.grid_mapping = 'cubed_sphere'
         q3.set_collective(True)

         q4 = ncfile.createVariable('q4', numpy.dtype('double').char, ('time',) + grid_data)
         q4.long_name = 'q4'
         q4.units = 'kg m-3'
         q4.standard_name = 'Tracer q4'
         q4.coordinates = 'lons lats'
         q4.grid_mapping = 'cubed_sphere'
         q4.set_collective(True)

   rank = MPI.COMM_WORLD.Get_rank()

   if rank == 0:
      xxx[:] = geom.x1[:]
      yyy[:] = geom.x2[:]
      if param.equations == "euler":
         # FIXME: With mapped coordinates, x3/height is a truly 3D coordinate
         zzz[:] = geom.x3[:,0,0] 

   tile[rank] = rank
   lon[rank,:,:] = geom.lon * 180/math.pi
   lat[rank,:,:] = geom.lat * 180/math.pi
   if param.equations == "euler":
      elev[rank,:,:,:] = geom.coordVec_latlon[2,:,:,:]
      topo[rank,:,:] = geom.zbot[:,:]


def output_netcdf(Q, geom, metric, mtrx, topo, step, param):
   """ Writes u,v,eta fields on every nth time step """
   rank = MPI.COMM_WORLD.Get_rank()
   idx = len(ncfile["time"])

   ncfile['time'][idx] = step * param.dt

   if param.equations == "shallow_water":

      # Unpack physical variables
      h = Q[idx_h, :, :] + topo.hsurf
      ncfile['h'][idx, rank, :, :] = h

      if param.case_number >= 2: # Shallow water
         u1 = Q[idx_hu1,:,:] / h
         u2 = Q[idx_hu2,:,:] / h
         u, v = contra2wind_2d(u1, u2, geom)
         rv = relative_vorticity(u1, u2, geom, metric, mtrx, param)
         pv = potential_vorticity(h, u1, u2, geom, metric, mtrx, param)

         ncfile['U'][idx, rank, :, :]  = u
         ncfile['V'][idx, rank, :, :]  = v
         ncfile['RV'][idx, rank, :, :] = rv
         ncfile['PV'][idx, rank, :, :] = pv

   if param.equations == "euler":
      rho   = Q[idx_rho, :, :, :]
      u1    = Q[idx_rho_u1, :, :, :]  / rho
      u2    = Q[idx_rho_u2, :, :, :]  / rho
      u3    = Q[idx_rho_w, :, :, :]   / rho
      theta = Q[idx_rho_theta, :,:,:] / rho

      u, v, w = contra2wind_3d(u1, u2, u3, geom, metric)

      ncfile['rho'][idx, rank, :,:,:]   = rho
      ncfile['U'][idx, rank, :, :]      = u
      ncfile['V'][idx, rank, :, :]      = v
      ncfile['W'][idx, rank, :, :]      = w
      ncfile['theta'][idx, rank, :,:,:] = theta
      ncfile['P'][idx, rank, :,:,:] = p0 * (Q[idx_rho_theta] * Rd / p0)**(cpd / cvd)

      if param.case_number == 11 or param.case_number == 12:
         ncfile['q1'][idx, rank, :,:,:] = Q[5, :,:,:] / rho

      if param.case_number == 11:
         ncfile['q2'][idx, rank, :,:,:] = Q[6, :,:,:] / rho
         ncfile['q3'][idx, rank, :,:,:] = Q[7, :,:,:] / rho
         ncfile['q4'][idx, rank, :,:,:] = Q[8, :,:,:] / rho

def output_finalize():
   """ Finalise the output netCDF4 file."""
   if MPI.COMM_WORLD.Get_rank() == 0:
      ncfile.close()
