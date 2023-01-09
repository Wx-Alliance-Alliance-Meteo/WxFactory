import numpy
import netCDF4
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
from math import ceil
import sys
from glob import glob
import re

def plot_field(dataFile, outputFile, idx = -1, field = 'h', nContour = 10, contoursLevels = None, error = False, lat_lims = (-90, 90), lon_lims = (-180,180), t = None, dataRefFile = None):
   # Load data
   data = netCDF4.Dataset(dataFile, 'r')

   # Setup figure
   fig = plt.figure(figsize=(6, 3))

   time = data['time'][:]
   if t is not None:
      idx = numpy.where(time == t * 3600 * 24)[0][0]

   lons = data['lons'][:].flatten()
   lats = data['lats'][:].flatten()
   vals = data[field][idx,...].flatten()

   if error:
      if dataRefFile is None:
         vals_ref = data[field][0,...].flatten()
      else:
         data_ref = netCDF4.Dataset(dataRefFile, 'r')
         vals_ref = data_ref[field][idx, ...].flatten()

      vals = numpy.maximum(numpy.finfo(float).eps, numpy.abs((vals - vals_ref) / vals_ref))

   delta_lat = 30
   delta_lon = 30

   selection = (lats >= 90 - delta_lat) & (lats <= 90)
   lons = numpy.concatenate((lons, (lons[selection] + 180) % 360))
   lats = numpy.concatenate((lats, 180 - lats[selection] ))
   vals = numpy.concatenate((vals, vals[selection]))

   selection = (lats >= -90) & (lats <= -90 + delta_lat)
   lons = numpy.concatenate((lons, (lons[selection] + 180) % 360))
   lats = numpy.concatenate((lats, -180 - lats[selection] ))
   vals = numpy.concatenate((vals, vals[selection]))

   lons[lons > 180] -= 360

   selection = (-180 <= lons) & (lons <= -180 + delta_lon)
   lons = numpy.concatenate((lons, lons[selection] + 360))
   lats = numpy.concatenate((lats, lats[selection]))
   vals = numpy.concatenate((vals, vals[selection]))

   selection = (180 - delta_lon <= lons) & (lons <= 180)
   lons = numpy.concatenate((lons, lons[selection] - 360))
   lats = numpy.concatenate((lats, lats[selection]))
   vals = numpy.concatenate((vals, vals[selection]))

   vmin = vals.min()
   vmax = vals.max()
   triang = tri.Triangulation(lons, lats)

   if contoursLevels:
      filled_c = plt.tricontourf(triang, vals, levels = contoursLevels, cmap='jet')
      #plt.tricontour(triang, vals, levels=contoursLevels, colors='k')
   elif error:
      #filled_c = plt.tricontourf(triang, vals, locator = ticker.LogLocator(), vmin = 1e-14, vmax = 2e-5, cmap='jet')
      filled_c = plt.tricontourf(triang, vals, locator=ticker.LogLocator(), cmap='jet')
   else:
      filled_c = plt.tricontourf(triang, vals, levels = numpy.linspace(vmin, vmax, nContour), cmap='jet')

   plt.xlim(lon_lims)
   plt.ylim(lat_lims)
   cbar = fig.colorbar(filled_c, orientation='vertical', shrink=1)
   #cbar.ax.tick_params(labelsize=45)
   fig.savefig(outputFile, bbox_inches='tight')
   plt.close()


def plot_quiver(dataFile, outputFile, idx = -1, nArrows = 15):
   # Load data
   data = netCDF4.Dataset(dataFile, 'r')
   nx = data['Xdim'].shape[0]
   ny = data['Ydim'].shape[0]
   npe = data['npe'].shape[0]
   vmin, vmax = data['h'][:, :, :, :].min(), data['h'][:, :, :, :].max()

   # Setup figure
   fig = plt.figure(figsize=(20, 30))
   ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
   ax.set_global()

   # Add colourful filled contours.
   stepX = ceil(nx / nArrows)
   stepY = ceil(ny / nArrows)

   for pe in range(npe):
      U = data['U'][idx, pe, ::stepX, ::stepY]
      V = data['V'][idx, pe, ::stepX, ::stepY]

      lats = data['lats'][pe, ::stepX, ::stepY]
      shift = 180 - lats[nArrows // 2, nArrows // 2]
      lats = (lats + shift) % 360 - shift

      lons = data['lons'][pe, ::stepX, ::stepY]
      shift = 180 - lons[nArrows // 2, nArrows // 2]
      lons = (lons + shift) % 360 - shift

      M = numpy.hypot(U, V)
      filled_c = plt.quiver(lons, lats, U, V, M, transform=ccrs.PlateCarree(), alpha=.8, units='inches', pivot='tip',
                            width=0.03, scale=1 / 0.015)

   cbar = fig.colorbar(filled_c, orientation='vertical', shrink=1)
   fig.savefig(outputFile, bbox_inches='tight')
   plt.close()


def plot_conservation(logFolder, outputFolder):
   orders = [3,4,5,6]
   for case in ['case2','case5','case6']:
      for match in ['mass', 'energy', 'enstrophy']:
         fig = plt.figure(figsize=(5, 3))
         for order in orders:
            print('log_' + case + '_order' + str(order) + '.ini')
            with open(logFolder + '/log_' + case + '_order' + str(order) + '.ini/1/rank.0/stdout') as of :
               content = of.read()
            m = re.findall('normalized integral of ' + match + ' = (.*)$', content, re.MULTILINE)
            val = list(map(float, m))
            plt.plot(numpy.arange(len(val)) * 900/(60*60*24), val)
            ax = plt.gca()
            ax.set_yscale('symlog', linthresh=1e-15)
            #ax.yaxis.get_major_formatter().set_powerlimits((0, 1))


         plt.legend(['Order ' + str(o) for o in orders], loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=5)
         plt.xlabel('Time (days)')

         fig.savefig(outputFolder + '/' + match + '_' + case + '.pdf', bbox_inches='tight')
         plt.close()

def plot_error(logFolder, outputFolder):
   orders = [3,4,5,6]
   for case in ['case2']:
      for match in ['l1', 'l2', 'linf']:
         fig = plt.figure(figsize=(5, 3))
         for order in orders:
            print('log_' + case + '_order' + str(order) + '.ini')
            with open(logFolder + '/log_' + case + '_order' + str(order) + '.ini/1/rank.0/stdout') as of :
               content = of.read()
            m = re.findall(match + ' = (.*?)\s', content, re.MULTILINE)
            val = list(map(float, m))
            plt.semilogy(numpy.arange(len(val)) * 900/(60*60*24), val)
            plt.xlabel('Time (days)')
            plt.ylabel('Error')

         plt.legend(['Order ' + str(o) for o in orders], loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5)
         fig.savefig(outputFolder + '/' + match + '_' + case + '.pdf', bbox_inches='tight')
         plt.close()

def plot_conv(logFolder, outputFolder, type):
   if type == 'ne':
      nvec = [10,12,14,16,18,20]
   else:
      nvec = [3,4,5,6,7,8]

   fig = plt.figure(figsize=(5, 3))
   for match in ['l1', 'l2', 'linf']:
      error = []
      for n in nvec:
         print('log_lauter_' + str(type) + '_' + str(n) + '.ini')
         with open(logFolder + '/log_lauter_' + str(type) + '_' + str(n) + '.ini/1/rank.0/stdout') as of:
            content = of.read()

         m = re.findall(match + ' = (.*?)\s', content, re.MULTILINE)
         error.append( float(m[-1]) )

      plt.semilogy(nvec, error)

   if type == 'ne':
      plt.xlabel('Number of elements')
   else:
      plt.xlabel('Solution point per element')

   plt.ylabel('Error')

   plt.legend(['$l_1$', '$l_2$', '$l_\infty$'])
   fig.savefig(outputFolder + '/' + 'lauther_' + type + '.pdf', bbox_inches='tight')
   plt.close()

def plot_res(dataFolder, plotFolder):
   files = glob(dataFolder + '/*.nc')
   patern = re.compile('.*/(\w+)_(\d+)x(\d+).nc')
   for f in files:
      m = patern.match(f)
      if m is not None:
         (case, resolution, order) = m.group(1, 2, 3)

         if case == 'case2':
            large_dt = False
            contoursLevels = []
            field = []
            lat_lims = (-90, 90)
            lon_lims = (-180, 180)
            time = [5]
         elif case == 'galewsky':
            large_dt = False
            contoursLevels = [list(numpy.arange(-1.5e-4, 1.8e-4, 2e-5))]
            field = ['RV']
            lat_lims = (0, 90)
            lon_lims = (-180, 180)
            time = [6]
         elif case == 'case6':
            large_dt = True
            contoursLevels = [list(range(8000, 10700, 100))]
            field = ['h']
            lat_lims = (-90, 90)
            lon_lims = (-180, 180)
            time = [14]
         else: # case 5
            large_dt = True
            contoursLevels = [list(range(5000,6050,50)), list(numpy.arange(-5e-5, 5e-5, 5e-6))]
            field = ['h', 'RV']
            lat_lims = (-90, 90)
            lon_lims = (-180, 180)
            time = [15, 7]

         name = case + '_' + str(resolution) + 'x' + str(order)
         dataFile = dataFolder + '/' + name + '.nc'
         print(name + '.nc')

         for i,f in enumerate(field):
            fieldFile = plotFolder + '/' + name + '_' + f + '.pdf'
            plot_field(dataFile, fieldFile, field=f, contoursLevels=contoursLevels[i], lat_lims = lat_lims, lon_lims = lon_lims, t = time[i])

         if case == 'case2':
            fieldFile = plotFolder + '/' + name + '_error_field.pdf'
            plot_field(dataFile, fieldFile, error=True, t = time[0])

         if large_dt:
            print(name + '_dt_4h.nc')
            dataFile_large_dt = dataFolder + '/' + name + '_dt_4h.nc'
            fieldFile = plotFolder + '/' + name + '_h_dt_4h.pdf'
            plot_field(dataFile_large_dt, fieldFile, contoursLevels=contoursLevels[0], lat_lims=lat_lims, lon_lims=lon_lims, t=time[0])
            fieldFile = plotFolder + '/' + name + '_error_h_dt_4h.pdf'
            plot_field(dataFile_large_dt, fieldFile, error=True, t=time[0], dataRefFile=dataFile)

         # windFile = plotFolder + '/' + name + '_wind.pdf'
         # plot_quiver(dataFile, windFile)

if __name__ == '__main__':
   if len(sys.argv) != 3:
      print('USAGE : python plot.py dataFolder plotFolder')
   else:
      plot_res(sys.argv[1], sys.argv[2])
      plot_conservation(sys.argv[1], sys.argv[2])
      plot_error(sys.argv[1], sys.argv[2])
      plot_conv(sys.argv[1], sys.argv[2], 'ne')
      plot_conv(sys.argv[1], sys.argv[2], 'solpts')

