from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_cubedsphere(geom):

   fig = plt.figure()
   ax = plt.axes(projection='3d')


   for pannel in range(6):
      ax.plot_surface(geom.cartX[:,:,pannel], geom.cartY[:,:,pannel], geom.cartZ[:,:,pannel], \
                      cmap='jet', edgecolor='none', shade=False)

#   ax.set_aspect('equal', 'box')

   plt.show()


def plot_field(geom, field):

   fig = plt.figure()
   ax = plt.axes(projection='3d')

   for pannel in range(6):
      ax.plot_surface(geom.cartX[:,:,pannel], geom.cartY[:,:,pannel], geom.cartZ[:,:,pannel], \
                      shade=False, facecolors=cm.jet(field[:,:,pannel]/field[:,:,pannel].max()) )

#   ax.set_aspect('equal', 'box')

   plt.show()
