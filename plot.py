import numpy
import netCDF4
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from math import ceil
import sys
from glob import glob
import re

def plot_field(dataFile, outputFile, idx = -1, contour = True, nContour = 10, showContour = False, labelContour = False):
   # Load data
   data = netCDF4.Dataset(dataFile, 'r')
   nx = data['Xdim'].shape[0]
   ny = data['Ydim'].shape[0]
   vmin, vmax = data['h'][:, :, :, :].min(), data['h'][:, :, :, :].max()

   # Setup figure
   fig = plt.figure(figsize=(20, 30))
   ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
   ax.set_global()

   for face in range(6):
      v = data['h'][idx, face, :, :]

      lats = data['lats'][face, :, :]
      shift = 180 - lats[nx // 2, ny // 2]
      lats = (lats + shift) % 360 - shift

      lons = data['lons'][face, :, :]
      shift = 180 - lons[nx // 2, ny // 2]
      lons = (lons + shift) % 360 - shift

      if contour :
         filled_c = plt.contourf(lons, lats, v, numpy.linspace(vmin, vmax, nContour), vmin=vmin, vmax=vmax,
                                 transform=ccrs.PlateCarree())
         if showContour:
            line_c = plt.contour(lons, lats, v, levels=filled_c.levels, colors=['black'], transform=ccrs.PlateCarree())
            if labelContour:
               plt.clabel(line_c, colors=['black'], manual=False, inline=True, fmt=' {:.0f} '.format)
      else :
         filled_c = plt.pcolormesh(lons, lats, v, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())


   ax.coastlines('110m', alpha=.1)
   fig.colorbar(filled_c, orientation='vertical', shrink=.3)
   fig.savefig(outputFile, bbox_inches='tight')
   plt.close()


def plot_quiver(dataFile, outputFile, idx = -1, nArrows = 15):
   # Load data
   data = netCDF4.Dataset(dataFile, 'r')
   nx = data['Xdim'].shape[0]
   ny = data['Ydim'].shape[0]
   vmin, vmax = data['h'][:, :, :, :].min(), data['h'][:, :, :, :].max()

   # Setup figure
   fig = plt.figure(figsize=(20, 30))
   ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
   ax.set_global()

   # Add colourful filled contours.
   stepX = ceil(nx / nArrows)
   stepY = ceil(ny / nArrows)

   for face in range(6):
      U = data['U'][idx, face, ::stepX, ::stepY]
      V = data['V'][idx, face, ::stepX, ::stepY]

      lats = data['lats'][face, ::stepX, ::stepY]
      shift = 180 - lats[nArrows // 2, nArrows // 2]
      lats = (lats + shift) % 360 - shift

      lons = data['lons'][face, ::stepX, ::stepY]
      shift = 180 - lons[nArrows // 2, nArrows // 2]
      lons = (lons + shift) % 360 - shift

      M = numpy.hypot(U, V)
      filled_c = plt.quiver(lons, lats, U, V, M, transform=ccrs.PlateCarree(), alpha=.8, units='inches', pivot='tip',
                            width=0.03, scale=1 / 0.015)

   ax.coastlines('110m', alpha=.1)
   fig.colorbar(filled_c, orientation='vertical', shrink=.3)
   fig.savefig(outputFile, bbox_inches='tight')
   plt.close()


def main(dataFolder, plotFolder):
   files = glob(dataFolder + '/*.nc')
   patern = re.compile('.*/(\w+)_(\d+)x(\d+).nc')
   for f in files:
      m = patern.match(f)
      if m is not None:
         (case, resolution, order) = m.group(1, 2, 3)
         print(case, resolution, order)
         name = case + '_' + str(resolution) + 'x' + str(order)
         dataFile = dataFolder + '/' + name + '.nc'

         fieldFile = plotFolder + '/' + name + '_field.pdf'
         plot_field(dataFile, fieldFile)

         fieldFile = plotFolder + '/' + name + '_field_1.pdf'
         plot_field(dataFile, fieldFile, showContour=True)

         fieldFile = plotFolder + '/' + name + '_field_2.pdf'
         plot_field(dataFile, fieldFile, showContour=True, labelContour=True)

         fieldFile = plotFolder + '/' + name + '_field_3.pdf'
         plot_field(dataFile, fieldFile, contour=False)


         windFile = plotFolder + '/' + name + '_wind.pdf'
         plot_quiver(dataFile, windFile)

if __name__ == '__main__':
   if len(sys.argv) != 3:
      print('USAGE : python plot.py dataFolder plotFolder')
   else:
      main(sys.argv[1], sys.argv[2])


