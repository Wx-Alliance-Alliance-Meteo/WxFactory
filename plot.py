import numpy
import netCDF4
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import ticker
from math import ceil, floor, sqrt
import sys
from glob import glob
import re

def plot_field(dataFile, outputFile, idx = -1, field = 'h', contour = True, nContour = 10, showContour = False, labelContour = False, contoursLevels = None, error = False):
   # Load data
   data = netCDF4.Dataset(dataFile, 'r')
   nx = data['Xdim'].shape[0]
   ny = data['Ydim'].shape[0]
   npe = data['npe'].shape[0]

   nb_pe_per_panel = npe // 6
   nb_lines_per_panel = int(sqrt(nb_pe_per_panel))
   nb_elems_per_line = nb_lines_per_panel

   nx_panel = nx * nb_elems_per_line
   ny_panel = ny * nb_lines_per_panel

   global_val = data[field][:,:,:,:]
   if error:
      global_val = numpy.maximum(numpy.finfo(float).eps, numpy.abs((global_val - global_val[0,:,:,:])/global_val[0,:,:,:]))
      # global_val = (global_val - global_val[0, :, :, :]) / global_val[0, :, :, :]

   vmin, vmax = global_val.min(), global_val.max()
   print(vmin, vmax)
   # Setup figure
   fig = plt.figure(figsize=(20, 30))
   ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
   ax.set_global()

   pe = 0
   for face in range(6):
      panel_lats = numpy.zeros((ny_panel, nx_panel))
      panel_lons = numpy.zeros((ny_panel, nx_panel))
      panel_vals = numpy.zeros((ny_panel, nx_panel))
      for idx_y in range(nb_lines_per_panel):
         for idx_x in range(nb_elems_per_line):
            panel_lats[idx_y*ny : (idx_y+1)*ny, idx_x*nx : (idx_x+1)*nx] = data['lats'][pe, :, :]
            panel_lons[idx_y*ny : (idx_y+1)*ny, idx_x*nx : (idx_x+1)*nx] = data['lons'][pe, :, :]
            panel_vals [idx_y * ny: (idx_y + 1) * ny, idx_x * nx: (idx_x + 1) * nx] = global_val[idx, pe, :, :]
            pe += 1

      shift = 180 - panel_lats[ny_panel//2, nx_panel//2]
      panel_lats = (panel_lats + shift) % 360 - shift

      shift = 180 - panel_lons[ny_panel//2, nx_panel//2]
      panel_lons = (panel_lons + shift) % 360 - shift

      if contour :
         if contoursLevels:
            filled_c = plt.contourf(panel_lons, panel_lats, panel_vals, contoursLevels, vmin=contoursLevels[0], vmax=contoursLevels[-1],
                                    transform=ccrs.PlateCarree(), cmap='jet')
         elif error:
            filled_c = plt.contourf(panel_lons, panel_lats, panel_vals, locator=ticker.LogLocator(), vmin=1e-12, vmax=2e-5,
                                    transform=ccrs.PlateCarree(), cmap='jet')
         else:
            filled_c = plt.contourf(panel_lons, panel_lats, panel_vals, numpy.linspace(vmin, vmax, nContour), vmin=vmin, vmax=vmax,
                                    transform=ccrs.PlateCarree(), cmap = 'jet')
         if showContour:
            line_c = plt.contour(panel_lons, panel_lats, panel_vals, levels=filled_c.levels, colors=['black'], transform=ccrs.PlateCarree())
            if labelContour:
               plt.clabel(line_c, colors=['black'], manual=False, inline=True, fmt=' {:.0f} '.format)
      else :
         filled_c = plt.pcolormesh(panel_lons, panel_lats, panel_vals, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

   cbar = fig.colorbar(filled_c, orientation='vertical', shrink=.3)
   cbar.ax.tick_params(labelsize=35)
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
   for case in ['case2','case5','case6','galewsky']:
      for match in ['mass', 'energy', 'enstrophy']:
         for order in orders:
            with open(logFolder + '/log_18x' + str(order) + '_' + case + '/1/rank.0/stdout') as of :
               content = of.read()
            m = re.findall('normalized integral of ' + match + ' = (.*)$', content, re.MULTILINE)
            val = list(map(float, m))
            plt.plot(numpy.arange(len(val)) * 900/(60*60*24), val)
            plt.xlabel('Time (days)')

         plt.legend(['Order ' + str(o) for o in [3,4,5,6]])
         plt.savefig(outputFolder + '/' + match + '_' + case + '.pdf', bbox_inches='tight')
         plt.close()

def plot_error(logFolder, outputFolder):
   orders = [3,4,5,6]
   for case in ['case2']:
      for match in ['l1', 'l2', 'linf']:
         for order in orders:
            with open(logFolder + '/log_18x' + str(order) + '_' + case + '/1/rank.0/stdout') as of :
               content = of.read()
            m = re.findall(match + ' = (.*?)\s', content, re.MULTILINE)
            val = list(map(float, m))
            plt.semilogy(numpy.arange(len(val)) * 900/(60*60*24), val)
            plt.xlabel('Time (days)')
            plt.ylabel('Error')

         plt.legend(['Order ' + str(o) for o in [3,4,5,6]], loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5)
         plt.savefig(outputFolder + '/' + match + '_' + case + '.pdf', bbox_inches='tight')
         plt.close()

def plot_conv(logFolder, outputFolder, type):
   if type == 'ne':
      nvec = [10,12,14,16,18,20]
   else:
      nvec = [3,4,5,6,7,8]

   print(nvec)
   for match in ['l1', 'l2', 'linf']:
      error = []
      for n in nvec:
         with open(logFolder + '/log_lauter_' + str(type) + '_' + str(n) + '/1/rank.0/stdout') as of:
            content = of.read()

         m = re.findall(match + ' = (.*?)\s', content, re.MULTILINE)
         error.append( float(m[-1]) )

      plt.semilogy(nvec, error)
      print (error)

   if type == 'ne':
      plt.xlabel('Number of elements')
   else:
      plt.xlabel('Solution point per element')

   plt.ylabel('Error')

   plt.legend(['$l_1$', '$l_2$', '$l_\infty$'])
   plt.savefig(outputFolder + '/' + 'lauther_' + type + '.pdf', bbox_inches='tight')
   plt.close()

def plot_res(dataFolder, plotFolder):
   files = glob(dataFolder + '/*.nc')
   patern = re.compile('.*/(\w+)_(\d+)x(\d+).nc')
   for f in files:
      m = patern.match(f)
      if m is not None:
         (case, resolution, order) = m.group(1, 2, 3)

         if case == 'case2':
            contoursLevels = list(range(1000,3200,200))
            field = 'h'
         elif case == 'galewsky':
            contoursLevels = list(numpy.arange(-1.5e-4, 1.7e-4, 2e-5))
            field = 'RV'
         elif case == 'case6':
            contoursLevels = list(range(8000, 10700, 100))
            field = 'h'
         else:
            contoursLevels = list(range(5000,6050,50))
            field = 'h'

         print(case, resolution, order)
         name = case + '_' + str(resolution) + 'x' + str(order)
         dataFile = dataFolder + '/' + name + '.nc'

         fieldFile = plotFolder + '/' + name + '_field.pdf'
         plot_field(dataFile, fieldFile, showContour=False, labelContour=False, contoursLevels=contoursLevels, field=field)

         if case == 'case2':
            fieldFile = plotFolder + '/' + name + '_error_field.pdf'
            plot_field(dataFile, fieldFile, showContour=False, labelContour=False, error=True)

         # windFile = plotFolder + '/' + name + '_wind.pdf'
         # plot_quiver(dataFile, windFile)

if __name__ == '__main__':
   if len(sys.argv) != 3:
      print('USAGE : python plot.py dataFolder plotFolder')
   else:
      plot_res(sys.argv[1]+'/model_output', sys.argv[2])

   plot_conservation(sys.argv[1], sys.argv[2])
   plot_error(sys.argv[1], sys.argv[2])

