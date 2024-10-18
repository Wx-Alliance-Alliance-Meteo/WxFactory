import math
import sys
import time
from typing import Callable, List

from mpi4py import MPI
import netCDF4
import numpy
from numpy.typing import NDArray

from common.definitions     import *
from common.configuration import Configuration
from geometry               import contra2wind_2d, contra2wind_3d
from output.diagnostic      import relative_vorticity, potential_vorticity

netcdf_serial = False


def prepare_array(param: Configuration) -> Callable[[NDArray], NDArray]:
   """
   Curried function to prepare an array for output:

   ```python
   # To prepare an array x for output:
   x = prepare_array(param)(x)
   ```

   If running with cuda configuration, this moves x to host memory.
   Otherwise, this function is a no-op
   """
   if param.device == "cuda":
      return lambda x: x.get()
   else:
      return lambda x: x

def store_field(field: NDArray, name: str, step_id: int, file: 'netCDF4.Dataset') -> None:
   """Store data in a given file.
   
   If the netcdf_serial option is activated, this will gather the data on a single PE, and
   only that PE will perform the write operation.
   """
   if netcdf_serial:
      fields: List = MPI.COMM_WORLD.gather(field, root=0)
      if MPI.COMM_WORLD.rank == 0:
         for i, f in enumerate(fields):
            file[name][step_id, i] = f
   else:
      file[name][step_id, MPI.COMM_WORLD.rank] = field

def output_init(geom, param):
   """ Initialise the netCDF4 file."""

   sys.stdout.flush()
   rank = MPI.COMM_WORLD.rank

   # creating the netcdf file(s)
   global ncfile
   ncfile = None

   try:
      ncfile = netCDF4.Dataset(param.output_file, 'w', format='NETCDF4', parallel = True)
   except ValueError:
      global netcdf_serial
      netcdf_serial = True
      if rank == 0:
         print(f'WARNING: Unable to open a netCDF4 file in parallel mode. Doing it serially instead')
         sys.stdout.flush()
         try:
            ncfile = netCDF4.Dataset(param.output_file, 'w', format='NETCDF4')
         except:
            print(f'unable to create file serially...')
            sys.stdout.flush()
            raise

   # create dimensions
   if param.equations == "shallow_water":
      ni, nj = geom.lat.shape
      grid_data = ('npe', 'Xdim', 'Ydim')
   elif param.equations == "euler":
      nk, nj, ni = geom.nk, geom.nj, geom.ni
      grid_data = ('npe', 'Zdim', 'Xdim', 'Ydim')
   else:
      raise ValueError(f"Unsupported equation type {param.equations}")

   grid_data2D = ('npe', 'Xdim', 'Ydim')

   if ncfile is not None:
      # write general attributes
      ncfile.history = 'Created ' + time.ctime(time.time())
      ncfile.description = 'GEF Model'
      ncfile.details = 'Cubed-sphere coordinates, Gauss-Legendre collocated grid'

      ncfile.createDimension('time', None) # unlimited
      npe = MPI.COMM_WORLD.Get_size()
      ncfile.createDimension('npe', npe)
      ncfile.createDimension('Ydim', ni)
      ncfile.createDimension('Xdim', nj)

      # create time axis
      tme = ncfile.createVariable('time', numpy.float64, ('time',))
      tme.units = 'hours since 1800-01-01'
      tme.long_name = 'time'
      tme.set_collective(not netcdf_serial)

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
         hhh.set_collective(not netcdf_serial)

         if param.case_number >= 2:
            uuu = ncfile.createVariable('U', numpy.dtype('double').char, ('time', ) + grid_data)
            uuu.long_name = 'eastward_wind'
            uuu.units = 'm s-1'
            uuu.standard_name = 'eastward_wind'
            uuu.coordinates = 'lons lats'
            uuu.grid_mapping = 'cubed_sphere'
            uuu.set_collective(not netcdf_serial)

            vvv = ncfile.createVariable('V', numpy.dtype('double').char, ('time', ) + grid_data)
            vvv.long_name = 'northward_wind'
            vvv.units = 'm s-1'
            vvv.standard_name = 'northward_wind'
            vvv.coordinates = 'lons lats'
            vvv.grid_mapping = 'cubed_sphere'
            vvv.set_collective(not netcdf_serial)

            drv = ncfile.createVariable('RV', numpy.dtype('double').char, ('time', ) + grid_data)
            drv.long_name = 'Relative vorticity'
            drv.units = '1/(m s)'
            drv.standard_name = 'Relative vorticity'
            drv.coordinates = 'lons lats'
            drv.grid_mapping = 'cubed_sphere'
            drv.set_collective(not netcdf_serial)

            dpv = ncfile.createVariable('PV', numpy.dtype('double').char, ('time', ) + grid_data)
            dpv.long_name = 'Potential vorticity'
            dpv.units = '1/(m s)'
            dpv.standard_name = 'Potential vorticity'
            dpv.coordinates = 'lons lats'
            dpv.grid_mapping = 'cubed_sphere'
            dpv.set_collective(not netcdf_serial)

      elif param.equations == "euler":
         elev = ncfile.createVariable('elev', numpy.dtype('double').char, grid_data)
         elev.long_name = 'Elevation'
         elev.units = 'm'
         elev.standard_name = 'Elevation'
         elev.coordinates = 'lons lats'
         elev.grid_mapping = 'cubed_sphere'
         elev.set_collective(not netcdf_serial)

         topo = ncfile.createVariable('topo', numpy.dtype('double').char, grid_data2D)
         topo.long_name = 'Topopgraphy'
         topo.units = 'm'
         topo.standard_name = 'Topography'
         topo.coordinates = 'lons lats'
         topo.grid_mapping = 'cubed_sphere'
         topo.set_collective(not netcdf_serial)

         uuu = ncfile.createVariable('U', numpy.dtype('double').char, ('time', ) + grid_data)
         uuu.long_name = 'eastward_wind'
         uuu.units = 'm s-1'
         uuu.standard_name = 'eastward_wind'
         uuu.coordinates = 'lons lats'
         uuu.grid_mapping = 'cubed_sphere'
         uuu.set_collective(not netcdf_serial)

         vvv = ncfile.createVariable('V', numpy.dtype('double').char, ('time', ) + grid_data)
         vvv.long_name = 'northward_wind'
         vvv.units = 'm s-1'
         vvv.standard_name = 'northward_wind'
         vvv.coordinates = 'lons lats'
         vvv.grid_mapping = 'cubed_sphere'
         vvv.set_collective(not netcdf_serial)

         www = ncfile.createVariable('W', numpy.dtype('double').char, ('time', ) + grid_data)
         www.long_name = 'upward_air_velocity'
         www.units = 'm s-1'
         www.standard_name = 'upward_air_velocity'
         www.coordinates = 'lons lats'
         www.grid_mapping = 'cubed_sphere'
         www.set_collective(not netcdf_serial)

         density = ncfile.createVariable('rho', numpy.dtype('double').char, ('time',) + grid_data)
         density.long_name = 'air_density'
         density.units = 'kg m-3'
         density.standard_name = 'air_density'
         density.coordinates = 'lons lats'
         density.grid_mapping = 'cubed_sphere'
         density.set_collective(not netcdf_serial)

         potential_temp = ncfile.createVariable('theta', numpy.dtype('double').char, ('time',) + grid_data)
         potential_temp.long_name = 'air_potential_temperature'
         potential_temp.units = 'K'
         potential_temp.standard_name = 'air_potential_temperature'
         potential_temp.coordinates = 'lons lats'
         potential_temp.grid_mapping = 'cubed_sphere'
         potential_temp.set_collective(not netcdf_serial)

         press = ncfile.createVariable('P', numpy.dtype('double').char, ('time',) + grid_data)
         press.long_name = 'air_pressure'
         press.units = 'Pa'
         press.standard_name = 'air_pressure'
         press.coordinates = 'lons lats'
         press.grid_mapping = 'cubed_sphere'
         press.set_collective(not netcdf_serial)

         if param.case_number == 11 or param.case_number == 12:
            q1 = ncfile.createVariable('q1', numpy.dtype('double').char, ('time',) + grid_data)
            q1.long_name = 'q1'
            q1.units = 'kg m-3'
            q1.standard_name = 'Tracer q1'
            q1.coordinates = 'lons lats'
            q1.grid_mapping = 'cubed_sphere'
            q1.set_collective(not netcdf_serial)

         if param.case_number == 11:
            q2 = ncfile.createVariable('q2', numpy.dtype('double').char, ('time',) + grid_data)
            q2.long_name = 'q2'
            q2.units = 'kg m-3'
            q2.standard_name = 'Tracer q2'
            q2.coordinates = 'lons lats'
            q2.grid_mapping = 'cubed_sphere'
            q2.set_collective(not netcdf_serial)

            q3 = ncfile.createVariable('q3', numpy.dtype('double').char, ('time',) + grid_data)
            q3.long_name = 'q3'
            q3.units = 'kg m-3'
            q3.standard_name = 'Tracer q3'
            q3.coordinates = 'lons lats'
            q3.grid_mapping = 'cubed_sphere'
            q3.set_collective(not netcdf_serial)

            q4 = ncfile.createVariable('q4', numpy.dtype('double').char, ('time',) + grid_data)
            q4.long_name = 'q4'
            q4.units = 'kg m-3'
            q4.standard_name = 'Tracer q4'
            q4.coordinates = 'lons lats'
            q4.grid_mapping = 'cubed_sphere'
            q4.set_collective(not netcdf_serial)

   prepare = prepare_array(param)

   if rank == 0:
      xxx[:] = prepare(geom.x1[:])
      yyy[:] = prepare(geom.x2[:])
      if param.equations == "euler":
         # FIXME: With mapped coordinates, x3/height is a truly 3D coordinate
         zzz[:] = prepare(geom.x3[:,0,0]) 

   if netcdf_serial:
      ranks = MPI.COMM_WORLD.gather(rank, root=0)
      lons  = MPI.COMM_WORLD.gather(prepare(geom.lon * 180/math.pi), root=0)
      lats  = MPI.COMM_WORLD.gather(prepare(geom.lat * 180/math.pi), root=0)
      if param.equations == "euler":
         elevs = MPI.COMM_WORLD.gather(prepare(geom.coordVec_latlon[2,:,:,:]), root=0)
         topos = MPI.COMM_WORLD.gather(prepare(geom.zbot[:,:]), root=0)

      if rank == 0:
         for my_rank, my_lon, my_lat in zip(ranks, lons, lats):
            tile[my_rank] = my_rank
            lon[my_rank, :, :] = my_lon
            lat[my_rank, :, :] = my_lat
         if param.equations == "euler":
            for my_rank, my_elev, my_topo in zip(ranks, elevs, topos):
               elev[my_rank, :, :, :] = my_elev
               topo[my_rank, :, :] = my_topo

   else:
      tile[rank] = rank
      lon[rank,:,:] = prepare(geom.lon * 180/math.pi)
      lat[rank,:,:] = prepare(geom.lat * 180/math.pi)
      if param.equations == "euler":
         elev[rank,:,:,:] = prepare(geom.coordVec_latlon[2,:,:,:])
         topo[rank,:,:] = prepare(geom.zbot[:,:])


def output_netcdf(Q, geom, metric, mtrx, topo, step, param):
   """ Writes u,v,eta fields on every nth time step """
   rank = MPI.COMM_WORLD.rank

   prepare = prepare_array(param)

   idx = 0
   if ncfile is not None:
      idx = len(ncfile['time'])
      ncfile['time'][idx] = step * param.dt

   if param.equations == "shallow_water":

      # Unpack physical variables
      h = Q[idx_h, :, :] + topo.hsurf
      store_field(prepare(h), 'h', idx, ncfile)

      if param.case_number >= 2: # Shallow water
         u1 = Q[idx_hu1,:,:] / h
         u2 = Q[idx_hu2,:,:] / h
         u, v = contra2wind_2d(u1, u2, geom)
         rv = relative_vorticity(u1, u2, geom, metric, mtrx, param)
         pv = potential_vorticity(h, u1, u2, geom, metric, mtrx, param)

         store_field(prepare(u), 'U', idx, ncfile)
         store_field(prepare(v), 'V', idx, ncfile)
         store_field(prepare(rv), 'RV', idx, ncfile)
         store_field(prepare(pv), 'PV', idx, ncfile)

   elif param.equations == "euler":
      rho   = Q[idx_rho, :, :, :]
      u1    = Q[idx_rho_u1, :, :, :]  / rho
      u2    = Q[idx_rho_u2, :, :, :]  / rho
      u3    = Q[idx_rho_w, :, :, :]   / rho
      theta = Q[idx_rho_theta, :,:,:] / rho

      u, v, w = contra2wind_3d(u1, u2, u3, geom, metric)

      store_field(prepare(rho), 'rho', idx, ncfile)
      store_field(prepare(u), 'U', idx, ncfile)
      store_field(prepare(v), 'V', idx, ncfile)
      store_field(prepare(w), 'W', idx, ncfile)
      store_field(prepare(theta), 'theta', idx, ncfile)
      store_field(prepare(p0 * (Q[idx_rho_theta] * Rd / p0)**(cpd / cvd)), 'P', idx, ncfile)

      if param.case_number == 11 or param.case_number == 12:
         store_field(prepare(Q[5, :, :, :] / rho), 'q1', idx, ncfile)

      if param.case_number == 11:
         for i in [6, 7, 8]:
            store_field(prepare(Q[i, :, :, :] / rho), f'q{i-4}', idx, ncfile)

def output_finalize():
   """ Finalise the output netCDF4 file."""
   if MPI.COMM_WORLD.rank == 0:
      ncfile.close()
