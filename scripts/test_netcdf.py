#!/usr/bin/env python3

import netCDF4

ncfile = netCDF4.Dataset('test.nc', 'w', format='NETCDF4', parallel = True)

